# Paired conformal method-specific recovery bounds

Status: prospectively locked feasibility study, separate from the completed
base campaign and every unpaired/natural-shift analysis. The machine-readable
configuration is `configs/paired_recovery_bounds/imagenet_c_clip_v1.json`.

## Object being bounded

For a 256-identity ImageNet-C batch, let `x_i` be a corrupted feature and let
`z_j` be a clean feature. The verified dataset identity supplies the true
bijection `pi*`, but evaluation-batch `pi*` is withheld from correction-map
fitting, cost construction, method predictions, and optimization. A frozen
correction method `a` predicts

`p_i^a = argmax h(M_a(x_i))`,

while the clean reference prediction is `r_j = argmax h(z_j)`. The study
bounds the method-specific clean-prediction recovery error

`R_a(pi) = n^{-1} sum_i 1[p_i^a != r_{pi(i)}]`.

This is not external target-label error. It is explicitly paired and
corruption-specific; it is neither a pure-marginal nor a natural-shift claim.

## One common feasible set

Identity, diagonal moment matching, and Bures correction are fitted once on
the same 4,096-identity map-fit endpoint sets. Pair assignments are erased
before all three fits. Independently of the method reward, a fourth role of
that same locked Bures/OAS map defines the preselected Play-3 geometry: full
reference-OAS Mahalanobis distance after Bures, divided by the full feature
dimension. Every method in a batch uses the identical cost matrix `C` and the
identical feasible set.

Let `C_min` be the minimum bijection cost and let `E C_rand = sum_ij C_ij/n`
be the exact mean cost of a uniformly random bijection. The score of a
bijection is

`S(pi) = (C(pi)-C_min)/(E C_rand-C_min)`.

The run fails closed if the denominator is not numerically positive. This
normalization reuses the existing random-minus-minimum cost scale and makes a
batch threshold interpretable as a share of the available random assignment
gap.

## Split-conformal statement

After an independent 4,096-pair map fit, the outcome-blind random partition
contains 29 calibration batches and 20 held-out evaluation batches per
corruption. Within every role, reference and target endpoint orders are
independently permuted by the locked RNG; the resulting true permutation is
withheld from evaluation cost construction, rewards, and optimization. For
`delta=0.1`, `m=29`, and

`k = ceil((m+1)(1-delta)) = 27`,

`qhat` is the 27th smallest calibration value of `S(pi*)`. In general, if
`k>m`, the implementation returns positive infinity exactly rather than
clipping the rank. Under exchangeability of the random batches conditional on
the independent map fit,

`P{S_new(pi*) <= qhat} >= 1-delta`.

On that event, `pi*` belongs to `F(qhat)={pi:S(pi)<=qhat}`. Consequently,
simultaneously for every predeclared method `a`,

`R_a(pi*) <= sup_{pi in F(qhat)} R_a(pi)`.

The simultaneity costs no methodwise Bonferroni factor: it follows from the one
shared feasibility event, not from three separately calibrated method events.
The guarantee concerns one future exchangeable batch within its corruption;
it is not a claim that all 20 evaluation batches are jointly covered.

## Optimization certificate and informativeness

For each method/batch, an integer-feasible assignment gives a lower bound `L`
on the supremum. For any `lambda>=0`, a linear-assignment solve gives

`U(lambda) = lambda B + max_pi {R_a(pi)-lambda C(pi)}`,

an upper bound for both the integer optimum and its assignment-LP relaxation.
The reported numerical upper bound `U` is the minimum across locked sampled
multipliers and the exact unconstrained label-count bound. Roundoff cushions
are recorded. These are reproducible floating-point optimization
certificates, not formally verified real-arithmetic proofs. A locked 16-row
HiGHS MILP diagnostic at relative gap zero checks containment for all methods
and corruptions.

Coverage alone cannot establish usefulness. Every batch therefore reports
`[L,U]`, width `U-L`, realized slack `U-R_a(pi*)`, the exact unconstrained
label-count bound `U_LC`, and certified tightening `U_LC-U`. Summaries retain
means and min--max ranges by corruption and method, plus true-pair feasibility
and simultaneous three-method bound coverage.

## Leakage and reporting boundary

- Feature-store manifests and every array are hash-verified before use.
- Head-fit identities are removed before the study partition.
- Partition selection uses authenticated opaque IDs only, never features,
  labels, logits, costs, predictions, or prior outcomes.
- Reference and target orderings are independently permuted, so source-table
  row alignment cannot act as an implicit correspondence hint.
- Map fits see paired endpoint membership but not pair assignment.
- Calibration uses true pairing only to score the common cost geometry.
- Evaluation bounds are frozen before the evaluation true pairing is revealed
  for feasibility and error scoring.
- Ground-truth class labels are never used. The clean frozen-readout prediction
  is the reference outcome.
- Gaussian noise and motion blur at severity 5 are a fixed two-condition
  feasibility panel, not a random corruption population.
- Missing units, failed exact diagnostics, invalid numerical gaps, or
  unauthenticated inputs make the study incomplete; no cell may be silently
  dropped or replaced.
