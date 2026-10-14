# Paired-recovery follow-up: prospective protocol

Status: **outcome-informed and frozen before any empirical follow-up output**.
The completed 2026-09-13 CLIP paired-recovery study and the 2026-09-14 audit
were read before this design. Synthetic arrays may be used for algebraic and
implementation tests; they are not study outcomes. No empirical output from
the partitions below may be inspected before the source lock is complete.

## Questions and reporting order

The primary question is whether the geometric assignment audit gives more
useful future-batch recovery bounds than direct calibration of recovery error
using the same fitting, calibration, and evaluation identities. The second
question is whether a common feasible correspondence set can certify signed
method differences. The third is whether informativeness depends on the
physical association encoded by the cost geometry. Split stability comes
fourth. Encoder breadth is explicitly secondary.

All outcomes are retained even if the direct baseline wins, no signed upper
bound is negative, or the column control remains informative. No axis may be
changed after inspecting a follow-up result.

## Frozen panel and information budget

The primary panel is CLIP on ImageNet-C Gaussian noise and motion blur at
severity 5 under three fresh partition seeds. Each encoder--corruption--split
cell uses 4,096 map-fit identities, 29 calibration batches, and 20 evaluation
batches of 256 identities. Roles are disjoint within a cell. Identities may
recur across randomized repetitions, which are stability checks on one fixed
dataset rather than new datasets.

ViT-B/16 is included for one matched partition per corruption only because
its native paired stores, frozen linear readout, exact array hashes, and
768-dimensional features are already present in the authenticated 306-store
manifest. Its split and control seeds equal CLIP repetition 1 so the identity
panels are matched. This was chosen on prerequisite and compute grounds, not
follow-up outcomes. DINOv3 is not part of this follow-up.

The complete grid has eight cells and 160 evaluation-batch units. A unit is
complete only when it contains all three absolute objectives and all three
directional signed contrasts under both physical and column-permuted geometry,
plus the three direct method bounds. There are 1,920 assignment certificates.

## Same-data direct recovery baseline

After identity, diagonal, and Bures maps are frozen from the assignment-erased
4,096-row map-fit endpoint sets, the map-fit role's true pairs are revealed
only for baseline standardization. Consecutive target positions form 16
256-identity error batches. For method `a`, fix

`c_a = mean(e_a,1,...,e_a,16)`

and

`s_a = max(sample_sd_ddof1(e_a,1,...,e_a,16), 1/256)`.

For each of the 29 calibration batches, compute

`z_j = max_a (e_a,j-c_a)/s_a`.

At `delta=0.1`, the threshold is the 27th ordered `z_j`. The direct upper
bound is `clip_[0,1](c_a+s_a*qhat_direct)`. The one shared score gives
simultaneous coverage across the three predeclared methods for one future
exchangeable batch conditional on maps fitted on the disjoint fit role. This
is a comparator, not a novel method. It is not a confidence interval for a
population mean.

"Same-data" means the raw paired-identity budget is matched: both approaches
receive the same fit, calibration, and evaluation identities. Their usable
information is not identical. The direct baseline uses true pairing on the
fit role to construct `c_a,s_a`, whereas geometric correction-map fitting
erases the fit-role assignment and sees only its two endpoint sets. This is a
deliberate advantage for the direct comparator and must be disclosed whenever
the comparison is reported.

## Signed common-correspondence contrasts

For each ordered directional contrast in the configuration, use the exact
edge reward

`w_ij = 1{p_i^a != r_j} - 1{p_i^c != r_j}`.

It remains in `{-1,0,1}` and is never clipped. The same integer-feasible and
Lagrangian assignment machinery brackets the maximum mean reward over the
same cost-feasible set used for absolute errors. A strictly negative upper
bound certifies lower recovery error for method `a` than method `c` throughout
that common set. Separate method upper bounds are not used to make this claim.

The directional contrasts are diagonal minus identity, Bures minus identity,
and Bures minus diagonal. They were fixed before follow-up outcomes.

## Column-permuted geometry control

For every calibration and evaluation batch, a locked keyed PCG64 draw
permutes only the columns of the cost matrix. Reference features, clean
predictions, reward matrices, and the true assignment retain their physical
indices. Transforming any of those with the columns would merely relabel the
original problem and is forbidden.

The permutation preserves the multiset of total costs over bijections, the
minimum assignment cost, and the exact uniform-random expected assignment
cost; the implementation checks the latter two invariants. The control has
its own threshold calibrated from its 29 permuted calibration scores. The
physical-geometry threshold is never reused.

The physical and control conformal events are separate 90% statements. No
joint 90% claim across them is made.

## Evaluation and claim boundaries

Evaluation true-assignment values are not consumed by cost construction,
method predictions, rewards, thresholds, or optimization. The deterministic
partition object necessarily contains them, and an opaque assignment hash may
be serialized during partition setup, but their values are first consumed
scientifically only after all bounds for a unit are frozen, to score
feasibility, actual method errors, actual contrasts, coverage, and slack.
Ground-truth ImageNet class labels are never used; recovery is relative to the
clean frozen-readout prediction.

For each cell and bound family report joint-across-method coverage, per-method
upper slack, and batch variation. For geometry report optimization width and
tightening. For contrasts report the actual signed difference, its
optimization bracket, its upper bound, and whether that upper bound is below
zero. For the column control report the separately calibrated threshold and
the same metrics. Wall-clock progress is operational monitoring only and is
not a prespecified scientific comparison between bound families. Across the
three CLIP repetitions report the distribution
of thresholds, coverage, slack, and certification counts, without treating
overlapping identities as independent datasets. ViT-B/16 is descriptive
breadth only. The two corruptions are fixed conditions, not a population.

Any missing or foreign unit; source, manifest, array, partition, or reward
hash mismatch; control-relabeling error; invalid direct scale; infeasible
primal witness; failed dual reconstruction; or failed exact diagnostic makes
the study incomplete. Nothing may be silently dropped or replaced.
