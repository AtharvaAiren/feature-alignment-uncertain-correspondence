"""Manifest-backed, memory-mapped endpoint feature stores.

The native V2 contract never infers correspondence from row order.  A store
may expose reference and target endpoints without exposing a pairing table.
Pairing is available only when separately hashed ID arrays and a verified
provenance declaration are present.

The legacy adapter reads the shared server layout
``<root>/<dataset>/<encoder>/<split>_{features,labels}.npy``.  It validates the
legacy per-file hashes and creates endpoint-local row IDs, but deliberately
does not claim cross-endpoint correspondence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    IntegrityError,
    describe_npy,
    read_json,
    seal_manifest,
    sha256_file,
    verify_file_record,
    verify_manifest_checksum,
    write_json_atomic,
)


NATIVE_MANIFEST = "FEATURE_STORE_MANIFEST.json"
NATIVE_SCHEMA = "marginal_alignment.feature_store.v2"
DEFAULT_LEGACY_ROOT = Path(os.environ.get("MARGINAL_LEGACY_FEATURE_ROOT", "data/legacy_features"))


@dataclass(frozen=True)
class Endpoint:
    name: str
    features: np.ndarray
    ids: np.ndarray
    labels: np.ndarray | None
    id_namespace: str
    id_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class DiscoveredFeatureStore:
    kind: str
    root: Path
    manifest_path: Path
    dataset: str
    encoder: str


@dataclass
class FeatureStore:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    kind: str
    dataset: str
    encoder: str
    records: Mapping[str, Mapping[str, Any]]
    endpoint_roles: Mapping[str, Mapping[str, str | None]]
    id_provenance: Mapping[str, Mapping[str, Any]]
    pairing_spec: Mapping[str, Any]
    hashes_verified: bool
    _generated_ids: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def array(self, role: str) -> np.ndarray:
        if role in self._generated_ids:
            return self._generated_ids[role]
        if role not in self.records:
            raise KeyError(f"feature-store role is absent: {role}")
        path = verify_file_record(self.root, self.records[role], verify_hash=False)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def endpoint(self, name: str) -> Endpoint:
        try:
            roles = self.endpoint_roles[name]
            provenance = self.id_provenance[name]
        except KeyError as error:
            raise KeyError(f"endpoint is absent: {name}") from error
        features = self.array(str(roles["features"]))
        ids = self.array(str(roles["ids"]))
        labels_role = roles.get("labels")
        labels = None if labels_role is None else self.array(str(labels_role))
        return Endpoint(
            name=name,
            features=features,
            ids=ids,
            labels=labels,
            id_namespace=str(provenance["namespace"]),
            id_provenance=provenance,
        )

    def pairing(self):
        """Return a validated PairingTable, never a row-order assumption."""

        from .split_plan import PairingTable

        if not self.hashes_verified:
            raise IntegrityError(
                "correspondence is disabled because feature-store hashes were not verified"
            )
        status = self.pairing_spec.get("status")
        if status != "available":
            reason = self.pairing_spec.get("reason", "no verified pairing was declared")
            raise IntegrityError(f"feature store has no usable correspondence: {reason}")
        provenance = self.pairing_spec.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("verified") is not True:
            raise IntegrityError("pairing provenance must be explicit and verified")
        if not provenance.get("kind") or not provenance.get("source"):
            raise IntegrityError("pairing provenance requires kind and source")
        reference = self.endpoint("reference")
        target = self.endpoint("target")
        reference_ids = self.array(str(self.pairing_spec["reference_ids_role"]))
        target_ids = self.array(str(self.pairing_spec["target_ids_role"]))
        return PairingTable(
            reference_ids=reference_ids,
            target_ids=target_ids,
            provenance=dict(provenance),
            reference_universe=reference.ids,
            target_universe=target.ids,
        )


def _ensure_rank_one_unique(values: np.ndarray, *, name: str) -> None:
    if values.ndim != 1:
        raise IntegrityError(f"{name} must be rank one; got {values.shape}")
    if len(np.unique(values)) != len(values):
        raise IntegrityError(f"{name} contains duplicate IDs")


def _validate_endpoint(store: FeatureStore, name: str) -> None:
    endpoint = store.endpoint(name)
    if endpoint.features.ndim != 2:
        raise IntegrityError(f"{name} features must have shape [samples,dimension]")
    _ensure_rank_one_unique(endpoint.ids, name=f"{name} IDs")
    if len(endpoint.features) != len(endpoint.ids):
        raise IntegrityError(f"{name} feature and ID row counts differ")
    if endpoint.labels is not None:
        if endpoint.labels.ndim != 1 or len(endpoint.labels) != len(endpoint.features):
            raise IntegrityError(f"{name} labels must have one value per feature row")
    provenance = endpoint.id_provenance
    if provenance.get("verified") is not True:
        raise IntegrityError(f"{name} ID provenance is not verified")
    if not provenance.get("kind") or not provenance.get("source") or not provenance.get("namespace"):
        raise IntegrityError(f"{name} ID provenance requires kind, source, and namespace")


def _validate_model_provenance(store: FeatureStore) -> None:
    """Require the encoder, preprocessing, layer, and head to be reproducible.

    Feature bytes alone do not identify the representation that produced
    them.  V2 therefore seals this information into every native store rather
    than accepting a directory name such as ``dinov3`` as provenance.
    """

    metadata = store.manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise IntegrityError("native feature-store metadata is malformed")
    model = metadata.get("model_provenance")
    if not isinstance(model, Mapping):
        raise IntegrityError("native feature store lacks model_provenance")
    required = {
        "encoder_id", "checkpoint_sha256", "preprocessing_sha256",
        "feature_layer", "pooling", "head_protocol", "head_fit_ids_sha256",
        "head_fit_ids_role", "head_fit_id_namespace", "reference_id_namespace",
        "reference_split", "head_fit_split",
    }
    missing = sorted(key for key in required if not model.get(key))
    if missing:
        raise IntegrityError(f"model_provenance is missing fields: {missing}")
    for key in ("checkpoint_sha256", "preprocessing_sha256", "head_fit_ids_sha256"):
        value = str(model[key])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise IntegrityError(f"model_provenance {key} is not a SHA-256")
    fit_role = str(model["head_fit_ids_role"])
    if fit_role not in store.records:
        raise IntegrityError(f"head-fit ID role is absent from arrays: {fit_role}")
    if str(store.records[fit_role]["sha256"]) != str(model["head_fit_ids_sha256"]):
        raise IntegrityError("head-fit ID hash disagrees with its sealed array record")
    _ensure_rank_one_unique(store.array(fit_role), name="head-fit IDs")
    reference = store.endpoint("reference")
    if str(model["reference_id_namespace"]) != reference.id_namespace:
        raise IntegrityError(
            "model_provenance reference_id_namespace disagrees with the reference endpoint"
        )
    head = store.manifest.get("head")
    if not isinstance(head, Mapping) or not head.get("weight_role") or not head.get("bias_role"):
        raise IntegrityError("native feature store lacks frozen-readout roles")
    for role in (str(head["weight_role"]), str(head["bias_role"])):
        if role not in store.records:
            raise IntegrityError(f"frozen-readout role is absent from arrays: {role}")


def build_native_manifest(
    root: str | Path,
    *,
    dataset: str,
    encoder: str,
    arrays: Mapping[str, str | Path],
    id_provenance: Mapping[str, Mapping[str, Any]],
    pairing: Mapping[str, Any] | None = None,
    head: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    destination: str | Path | None = None,
) -> Path:
    """Seal existing NPY arrays into the V2 feature-store contract."""

    base = Path(root).expanduser().resolve()
    required = {"reference_features", "reference_ids", "target_features", "target_ids"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"native feature store is missing roles: {missing}")
    records: dict[str, dict[str, Any]] = {}
    for role, value in sorted(arrays.items()):
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        try:
            path.relative_to(base)
        except ValueError as error:
            raise IntegrityError(f"array for role {role} is outside the store: {path}") from error
        records[str(role)] = describe_npy(path, relative_to=base)
    pairing_payload: Mapping[str, Any] = pairing or {
        "status": "unavailable",
        "reason": "no cross-endpoint pairing was supplied",
    }
    payload = {
        "schema_version": NATIVE_SCHEMA,
        "dataset": str(dataset),
        "encoder": str(encoder),
        "arrays": records,
        "endpoints": {
            "reference": {
                "features": "reference_features",
                "ids": "reference_ids",
                "labels": "reference_labels" if "reference_labels" in records else None,
            },
            "target": {
                "features": "target_features",
                "ids": "target_ids",
                "labels": "target_labels" if "target_labels" in records else None,
            },
        },
        "id_provenance": dict(id_provenance),
        "pairing": dict(pairing_payload),
        "head": dict(head or {}),
        "metadata": dict(metadata or {}),
    }
    manifest = seal_manifest(payload)
    output = base / NATIVE_MANIFEST if destination is None else Path(destination)
    if not output.is_absolute():
        output = base / output
    write_json_atomic(output, manifest)
    # A manifest is not published successfully unless the full contract loads.
    load_feature_store(output, verify_hashes=True)
    return output


def _load_native(path: Path, *, verify_hashes: bool) -> FeatureStore:
    manifest = read_json(path)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != NATIVE_SCHEMA:
        raise IntegrityError(f"not a {NATIVE_SCHEMA} manifest: {path}")
    verify_manifest_checksum(manifest)
    records = manifest.get("arrays")
    endpoints = manifest.get("endpoints")
    provenance = manifest.get("id_provenance")
    if not isinstance(records, Mapping) or not isinstance(endpoints, Mapping):
        raise IntegrityError("native feature-store arrays/endpoints are malformed")
    if not isinstance(provenance, Mapping):
        raise IntegrityError("native feature store lacks ID provenance")
    root = path.parent.resolve()
    for role, record in records.items():
        if not isinstance(record, Mapping):
            raise IntegrityError(f"array record is malformed: {role}")
        if record.get("format") != "npy":
            raise IntegrityError(f"native role {role} must be an NPY artifact")
        verify_file_record(root, record, verify_hash=verify_hashes)
    store = FeatureStore(
        root=root,
        manifest_path=path.resolve(),
        manifest=manifest,
        kind="native_v2",
        dataset=str(manifest.get("dataset", "")),
        encoder=str(manifest.get("encoder", "")),
        records=records,
        endpoint_roles=endpoints,
        id_provenance=provenance,
        pairing_spec=manifest.get("pairing", {}),
        hashes_verified=verify_hashes,
    )
    if not store.dataset or not store.encoder:
        raise IntegrityError("native feature store requires dataset and encoder")
    _validate_endpoint(store, "reference")
    _validate_endpoint(store, "target")
    _validate_model_provenance(store)
    if store.endpoint("reference").features.shape[1] != store.endpoint("target").features.shape[1]:
        raise IntegrityError("reference and target feature dimensions differ")
    if store.pairing_spec.get("status") == "available":
        if verify_hashes:
            store.pairing()
        else:
            for key in ("reference_ids_role", "target_ids_role", "provenance"):
                if key not in store.pairing_spec:
                    raise IntegrityError(f"available pairing declaration lacks {key}")
    elif store.pairing_spec.get("status") != "unavailable":
        raise IntegrityError("pairing status must be available or unavailable")
    return store


def _legacy_file_records(root: Path, manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("verified") is not True:
        raise IntegrityError(f"legacy cache is not marked verified: {root}")
    files = manifest.get("files")
    if not isinstance(files, (Mapping, list)):
        raise IntegrityError("legacy manifest needs per-file entries under 'files'")
    entries: list[tuple[str, Any]]
    if isinstance(files, Mapping):
        entries = [(str(name), value) for name, value in files.items()]
    else:
        entries = []
        for value in files:
            if not isinstance(value, Mapping) or not value.get("path"):
                raise IntegrityError("legacy file-list entries require a path")
            entries.append((str(value["path"]), value))
    records: dict[str, Mapping[str, Any]] = {}
    for name, specification in entries:
        if isinstance(specification, str):
            specification = {"sha256": specification}
        if not isinstance(specification, Mapping):
            raise IntegrityError(f"legacy artifact declaration is malformed: {name}")
        relative = str(specification.get("path", name))
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"legacy artifact escapes cache root: {relative}") from error
        if not path.is_file():
            raise IntegrityError(f"legacy artifact is missing: {path}")
        expected = specification.get("sha256")
        if not isinstance(expected, str):
            raise IntegrityError(f"legacy artifact lacks an exact SHA-256: {relative}")
        inspected = describe_npy(path, relative_to=root)
        inspected["sha256"] = expected
        records[path.name] = inspected
    return records


def _load_legacy(path: Path, *, verify_hashes: bool) -> FeatureStore:
    root = path.parent.resolve()
    manifest = read_json(path)
    if not isinstance(manifest, Mapping):
        raise IntegrityError(f"legacy manifest is malformed: {path}")
    records_by_file = _legacy_file_records(root, manifest)
    for record in records_by_file.values():
        verify_file_record(root, record, verify_hash=verify_hashes)
    feature_names = sorted(name for name in records_by_file if name.endswith("_features.npy"))
    if not feature_names:
        raise IntegrityError(f"legacy cache exposes no *_features.npy files: {root}")
    records: dict[str, Mapping[str, Any]] = {}
    endpoints: dict[str, Mapping[str, str | None]] = {}
    provenance: dict[str, Mapping[str, Any]] = {}
    generated: dict[str, np.ndarray] = {}
    dataset = str(manifest.get("dataset") or root.parent.name)
    encoder = str(manifest.get("encoder") or root.name)
    for feature_name in feature_names:
        split = feature_name[: -len("_features.npy")]
        feature_role = f"{split}_features"
        id_role = f"{split}_ids"
        label_file = f"{split}_labels.npy"
        if label_file not in records_by_file:
            raise IntegrityError(f"legacy endpoint {split} lacks hashed labels")
        label_role = f"{split}_labels"
        records[feature_role] = records_by_file[feature_name]
        records[label_role] = records_by_file[label_file]
        feature_path = verify_file_record(root, records[feature_role], verify_hash=False)
        count = int(np.load(feature_path, mmap_mode="r", allow_pickle=False).shape[0])
        generated[id_role] = np.arange(count, dtype=np.int64)
        namespace = f"legacy:{dataset}:{encoder}:{split}"
        endpoints[split] = {"features": feature_role, "ids": id_role, "labels": label_role}
        provenance[split] = {
            "verified": True,
            "kind": "endpoint_local_row_index",
            "source": str(path),
            "namespace": namespace,
            "cross_endpoint_correspondence": False,
        }
    store = FeatureStore(
        root=root,
        manifest_path=path.resolve(),
        manifest=manifest,
        kind="legacy_server_cache",
        dataset=dataset,
        encoder=encoder,
        records=records,
        endpoint_roles=endpoints,
        id_provenance=provenance,
        pairing_spec={
            "status": "unavailable",
            "reason": "legacy row order is endpoint-local and is not correspondence provenance",
        },
        hashes_verified=verify_hashes,
        _generated_ids=generated,
    )
    for endpoint in endpoints:
        _validate_endpoint(store, endpoint)
    return store


def load_feature_store(path: str | Path, *, verify_hashes: bool = True) -> FeatureStore:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        native = candidate / NATIVE_MANIFEST
        legacy = candidate / "manifest.json"
        if native.is_file():
            candidate = native
        elif legacy.is_file():
            candidate = legacy
        else:
            raise FileNotFoundError(f"no feature-store manifest under {candidate}")
    if candidate.name == NATIVE_MANIFEST:
        return _load_native(candidate, verify_hashes=verify_hashes)
    if candidate.name == "manifest.json":
        return _load_legacy(candidate, verify_hashes=verify_hashes)
    raise IntegrityError(f"unrecognized feature-store manifest name: {candidate}")


def _discovery_roots(roots: Sequence[str | Path] | None) -> list[Path]:
    if roots is not None:
        candidates = [Path(value).expanduser() for value in roots]
    else:
        configured = os.environ.get("MARGINAL_ALIGNMENT_FEATURE_ROOTS", "")
        candidates = [Path(value).expanduser() for value in configured.split(os.pathsep) if value]
        candidates.append(DEFAULT_LEGACY_ROOT)
    return sorted({path.resolve() for path in candidates if path.is_dir()})


def discover_feature_stores(
    roots: Sequence[str | Path] | None = None,
) -> list[DiscoveredFeatureStore]:
    """Discover native and shared-server stores without assuming dataset names."""

    found: dict[Path, DiscoveredFeatureStore] = {}
    for root in _discovery_roots(roots):
        for path in sorted(root.rglob(NATIVE_MANIFEST)):
            try:
                manifest = read_json(path)
            except IntegrityError:
                continue
            if manifest.get("schema_version") != NATIVE_SCHEMA:
                continue
            store_root = path.parent.resolve()
            found[store_root] = DiscoveredFeatureStore(
                kind="native_v2",
                root=store_root,
                manifest_path=path.resolve(),
                dataset=str(manifest.get("dataset", store_root.parent.name)),
                encoder=str(manifest.get("encoder", store_root.name)),
            )
        for path in sorted(root.rglob("manifest.json")):
            store_root = path.parent.resolve()
            if store_root in found or not any(store_root.glob("*_features.npy")):
                continue
            try:
                manifest = read_json(path)
            except IntegrityError:
                continue
            found[store_root] = DiscoveredFeatureStore(
                kind="legacy_server_cache",
                root=store_root,
                manifest_path=path.resolve(),
                dataset=str(manifest.get("dataset") or store_root.parent.name),
                encoder=str(manifest.get("encoder") or store_root.name),
            )
    return [found[key] for key in sorted(found, key=str)]
