# Decision Tree criteria and stopping parameters

## Objective and authorized scope

Extend the custom teaching Decision Tree with an easy Information Gain / Gain
Ratio switch, maximum depth, and minimum samples for splitting. Information Gain
remains active; the adjacent Gain Ratio assignment is commented out and uses the
same score variable. T3 authorizes replacing the original node-local minimum with
root-weighted Information Gain, while keeping attribute ranking node-local.

- Edit `decision-tree.ipynb`, `test_decision_tree.py`, and this document only.
- Do not modify Random Forest, its tests, requirements, or unrelated work.
- No dependency installation, remote execution, push, pull request, or merge.
- Local tracker branch: `feat/dt-criteria-stopping`; starting boundary: `b524d86`.

## Decisions and acceptance criteria

- Normalize Gain Ratio by the entropy of branch proportions; handle zero safely.
- Rank attributes by local Information Gain or local Gain Ratio; keep the
  existing assignment pair and first-attribute tie behavior unchanged.
- Keep `min_gain=1e-12`, but apply it to the selected attribute's
  `weighted_gain = len(D) / n_total * (p0 - pg)`. Stop only for `< min_gain`;
  equality permits splitting. Never normalize this minimum by Gain Ratio.
- Initialize optional `n_total=None` once from the original training `len(D)`;
  every child receives that same original total, not its own or its parent's size.
- Append `n_total` after the six existing arguments; preserve their positional
  compatibility, all other defaults, and the existing direct recursion.
- Use unit row counts only; no sample weights, RF changes, nested builder,
  numerical split search, or validation/configuration framework are authorized.
- Root depth is zero; `max_depth=None` is unlimited; zero creates a majority leaf.
- `min_samples_split=2` by default; fewer samples create a majority leaf.
- Propagate both stopping settings through direct recursion.
- Test both criterion lines by switching copied source in memory, not on disk.
- Preserve notebook cell IDs and saved outputs; avoid unrelated JSON churn.
- Reconcile the stale stdout assertion with existing predictions and accuracy
  output in the DT test harness; do not change runtime output to hide the failure.

The local-only minimum described in historical T1/T2 evidence was correct for
those commits. The user explicitly accepted its replacement in T3; that history
does not override the new active minimum contract or reopen completed controls.

## Testing configuration and baseline

- TDD: **off for this feature**, explicitly selected by the user as `Sin TDD`.
  Source: the original native user answer, Engram observation `4939`, and this
  tracker. This carries into T3; existing tests do not establish another mode.
- Runner: `unittest`, Python 3.14.7; ordinary functional tests remain required.
- Focused: `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_decision_tree -v`.
- Full: `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -v`.
- Baseline focused: 13 tests, 12 passed, one stale stdout assertion failed:
  `test_final_cell_reports_only_held_out_accuracy` expects only accuracy, while
  the existing notebook also prints the predictions list.
- Baseline full: the same failure plus `test_random_forest` import error because
  `sklearn` is absent. No installation or RF change is authorized.
- Final structural checks: notebook JSON, cell identity preservation, changed
  path scope, `git diff --check`, and unchanged RF files.

## Work units

- [x] **T1 — Criterion switch:** add normalized Gain Ratio alongside active
  Information Gain, preserve raw minimum gain, add behavior tests and notebook
  explanations, and reconcile the stale DT stdout expectation.
  Route: delegated direct; notebook + tests are two non-trivial files, and
  preparation belongs to the same bounded writer. Commit: `74ea6b6`.
- [x] **T2 — Stopping parameters:** add depth and minimum-sample controls,
  recursive propagation, boundary tests, and usage documentation.
  Route: delegated direct; same two-file writer trigger. Commit: `7eb4f0a`.
- [ ] **T3 — Root-weighted minimum:** initialize/propagate original training N,
  apply the weighted selected-gain threshold, revise local-only regression and
  notebook explanations, and retain ranking/control/output regressions.
  Route: delegated direct; notebook and tests are two non-trivial files, and
  preparation belongs to the same bounded writer. Commit: pending.
  Parent read both preparation artifacts and authorized source implementation.
  Implementation checks are recorded below; verification/commit remain parent-owned.

## Delivery and review evidence

- Initial T1/T2 authored forecast: T1 250–340 lines; T2 170–240; total 420–580 additions
  plus deletions, including tracking and excluding saved-output churn.
- About 400 lines per task is advisory, not a reason to omit tests or minify.
- Delivery strategy: `ask-on-risk`; user selected `feature-branch-chain`.
- Local tracker: `feat/dt-criteria-stopping`; T1: `feat/dt-criteria` from tracker;
  T2: `feat/dt-stopping-parameters` from T1; T3: `feat/dt-weighted-min-gain` from
  T2's doc-closeout boundary `f9f89a9`. No PRs, push, or merges authorized.
- Running committed authored count: 619 through `f9f89a9`: implementation 545
  (T1: 270 additions + 34 deletions = 304; T2: 221 + 20 = 241, including T1
  tracking updates) plus the separately accounted doc-only closeout 60 + 14 = 74.
- The documentation-only closeout was committed as `f9f89a9`, not left pending.
  Parent assessment was passive/non_executable_only, `review_due=false`.
- RDD: on, deciding source default. Native assessment/review is parent-owned.
- Last reviewed implementation boundary: `7eb4f0a`. Both implementation candidates
  were independently assessed high, explicitly consented, four-lens reviewed,
  and acknowledged with authority burned by the parent. No correction was required.
- Rollback: revert each work-unit commit without touching RF or dependencies.

## Progress and next step

T1 is complete on `feat/dt-criteria` at `74ea6b6`; the parent spot check passed
all 40 DT tests. Native four-lens review found no issues and acknowledgement
confirmed authority burned for lineage `review-ecd389460b7f5458`.
T2 is complete on `feat/dt-stopping-parameters` at `7eb4f0a`; the parent independently
verified its structure, repeated all 54 DT tests, and completed its native review
and acknowledgement. Both implementation tasks are complete. Code is ready on
the current dependent branch, not merged into `main` or the local tracker branch.
The documentation-only closeout is complete at `f9f89a9`. After both preparation
readbacks, parent authorized T3 source implementation on `feat/dt-weighted-min-gain`.
The weighted minimum is now implemented and all 62 DT tests pass. Full discovery
has only the known RF import error. T3 remains unchecked and uncommitted pending
parent verification; its separate native assessment/consent is still parent-owned.
T1/T2 implementation evidence below remains historical and valid for its boundaries.

### T1 implementation

- Added branch-proportion entropy in `split_information(Ai, D)`, returning zero
  for empty data and constant partitions.
- Kept adjacent `score` assignments: Information Gain active, Gain Ratio commented.
  Maximum score selects the attribute; ties retain the first attribute in `A`.
- Preserved `decision_tree(D, A, min_gain=1e-12)` and direct recursive calls.
  The selected attribute's raw node-local gain controls the strict minimum;
  no normalized/root-weighted threshold or substitute-attribute filter was added.
- Both criterion settings run the shared regression suite using copied source
  in fresh namespaces. Tests cover differing rankings, normalization, zero
  denominators, ties, local/raw thresholds, and existing categorical behavior.
- Normalization completed before checks: renamed the stale stdout test and
  asserted the full existing accuracy-plus-predictions output. Runtime output
  code and saved outputs were not changed; no formatter was installed or run.

### T1 check evidence

- `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_decision_tree -v`:
  PASS, 40 tests in 0.034s, covering both criterion variants.
- `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -v`:
  41 reported tests in 0.033s, 40 passed, one known baseline import error:
  `unittest.loader._FailedTest.test_random_forest`, caused by
  `ModuleNotFoundError: No module named 'sklearn'` at `test_random_forest.py:12`.
  No DT failures; RF tests and notebook did not execute. No install attempted.
- `git diff --check`: PASS, no reported whitespace errors.
- `git diff --exit-code b524d86 -- random-forest.ipynb test_random_forest.py requirements.txt`:
  PASS, no differences.
- Read-only JSON/AST comparison against `b524d86`: PASS, 11 cells and five code
  cells; identical cell IDs, metadata, execution counts, saved outputs, training
  signature, majority helper, and prediction/display helpers. Source changes are
  limited to introduction, impurity/algorithm/accuracy explanations, impurity
  helpers, and attribute scoring. No persisted execution outputs or bytecode.
- Runtime harness: the focused command executes all five cells in memory for
  both settings. No external-access attempt was observed during these checks.

### T1 boundary and next step

- Changed paths: `decision-tree.ipynb`, `test_decision_tree.py`, and this tracker.
- Rollback boundary: remove T1 notebook/test changes and this progress entry;
  preserve parent tracking decisions, T2, RF, and dependencies.
- The T1 writer performed no staging, commits, branch changes, reviews, remote
  operations, or delegation. The parent committed and reviewed the exact unit.
- Full tracker mirror target: Engram observation `4940`, topic
  `odd/decision-tree-criteria-stopping/tasks`.
- Committed boundary: `b524d86..74ea6b6`; T1 includes notebook, tests, and docs.
- Parent repeated the focused command: PASS, 40 tests in 0.034s.
- Native committed assessment: high (`process_boundary`), review was due; user granted
  this candidate's review. All four lenses reported no findings. Exact native
  acknowledgement returned `action=acknowledged` and `authority=burned`.
- Earlier uncommitted assessment was unassessable due to the untracked tracker;
  it did not lower risk. Committed assessment resolved the inventory normally.
- Historical handoff after T1: proceed to T2; no push, PR, merge, or main change
  had occurred at that boundary.

### T2 implementation

- Added `decision_tree(D, A, min_gain=1e-12, max_depth=None,
  min_samples_split=2, depth=0)`, preserving the original positional arguments.
- The root starts at depth zero. A reached/exceeded maximum returns the current
  node's majority leaf; `None` imposes no explicit depth limit and zero stops
  the root. The sample guard stops when the current `len(D)` is strictly below
  its minimum; equality permits splitting and this is not a minimum leaf size.
- Direct recursion forwards both limits unchanged and increments `depth` by one.
  T1 criterion assignments, scoring, raw-gain comparison, ties, and categorical
  partitions are unchanged; no nested builder or validation framework was added.
- The notebook training call exposes `max_depth=None, min_samples_split=2`.
  Its documentation explains changing those values, for example to `3` and `5`,
  without changing the `depth=0` recursion bookkeeping default.
- Added behavior tests for zero/one/multiple/unlimited depths, exceeded caps,
  descendant majority leaves differing from the root, sample boundaries and
  child-local propagation, smaller leaves, combined stops, and positional gain.
  Both existing criterion variants run all tests; AST checks enforce forwarding.
- Normalization: no additional formatter/cleanup was needed or run. Source edits
  used the existing notebook JSON layout before final checks; the complete T1
  accuracy-plus-predictions assertion and all saved outputs remain unchanged.

### T2 check evidence

- `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_decision_tree -v`:
  PASS, 54 tests in 0.050s, covering both criterion variants.
- `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -v`:
  55 reported tests in 0.049s, 54 passed, only the known baseline import error
  `unittest.loader._FailedTest.test_random_forest` due to
  `ModuleNotFoundError: No module named 'sklearn'` at `test_random_forest.py:12`.
  No DT failures; no RF notebook execution or dependency installation attempted.
- `git diff --check`: PASS, no reported whitespace errors.
- `git diff --exit-code b524d86 -- random-forest.ipynb test_random_forest.py requirements.txt`:
  PASS, no differences.
- Read-only JSON/AST comparison against `74ea6b6`: PASS, 11 cells/five code cells;
  all cell IDs, metadata, execution counts, and saved outputs unchanged. Only
  `algorithm-description` and `c45-algorithm` source changed. T1 scoring/raw-gain
  source and majority/prediction/display helpers are unchanged.
- Independent in-memory cell execution against `74ea6b6`: PASS, complete default
  example stdout matches T1 exactly under both criterion variants. No notebook
  execution outputs or bytecode were persisted; no external access was observed.

### T2 boundary and next step

- Changed paths remain the notebook, DT tests, and this tracker. Preserve all
  inherited parent tracking edits and the complete T1 evidence in this unit.
- Rollback boundary: restore T2 notebook/tests to `74ea6b6` and remove only T2
  progress text; retain parent T1 completion/review updates, RF, and dependencies.
- During implementation, the T2 writer performed no staging, commits, branch
  changes, native review, delegation, installation, remote operations, or delivery.
  These writer constraints do not describe the parent's subsequent local actions.
- The original handoff left parent verification, commit, and independent native
  review pending; the parent has now completed all three without correction.
- Committed boundary: `74ea6b6..7eb4f0a`; full T2 commit:
  `7eb4f0a58cad26899e2ab54917313cb4f5817a76`.
- Parent structural diff check: PASS. Parent independently repeated
  `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_decision_tree -v`:
  PASS, 54 tests in 0.053s. No other failures or skipped DT checks were reported.
- Parent-recorded native committed assessment: high (`process_boundary` /
  `shell_process` in `test_decision_tree.py`). The user explicitly granted this
  T2 candidate's review; all four lenses reported zero findings.
- Exact native acknowledgement for lineage `review-6eb8dbdb78a1d773` returned
  schema `gentle-ai.review-acknowledged/v1`, `action=acknowledged`, and
  `authority=burned`. No correction was required.

### T1/T2 closeout snapshot and current branch state

- Implementation tasks T1 and T2 are complete. `feat/dt-stopping-parameters`
  contains both units plus doc-only closeout `f9f89a9`; `feat/dt-criteria` holds T1
  at `74ea6b6`. Current branch `feat/dt-weighted-min-gain` starts at `f9f89a9`.
  The user-selected feature-branch-chain slices and prior implementation are intact.
- `main` and the local tracker branch `feat/dt-criteria-stopping` remain at
  `b524d86`. No merges, pushes, PRs, or remote operations occurred. Implementation
  readiness on the dependent branch is not delivery into either base branch.
- Current implemented API: `decision_tree(D, A, min_gain=1e-12, max_depth=None,
  min_samples_split=2, depth=0, n_total=None)`. Information Gain is active; the
  adjacent Gain Ratio assignment is commented, with both assigning `score`.
  T3 weights only the minimum by the node's fraction of original training rows;
  local ranking is unchanged. Root depth is zero; `None` is unlimited and the
  sample minimum is strict and node-local, not a minimum leaf size.
- On the current branch, open `decision-tree.ipynb` and edit the adjacent criterion
  assignments or the `max_depth` / `min_samples_split` values in the training call
  of `c45-algorithm`, then rerun the notebook cells. Keep `depth=0` as bookkeeping.
- The last full-suite limitation remains the pre-existing missing-`sklearn` RF
  import error: 62 DT passes plus one RF import error. No installation was
  authorized or performed; there are no other failures or skipped DT checks.
- The earlier recovery changed only this tracker and full mirror `4940`; no
  source/test edits or reruns occurred during that bookkeeping phase. The new T3
  focused baseline is separately recorded below, without editing source/tests.
- Doc-only closeout commit: `f9f89a9376f8496adb2f3d3e85b3738983e5667f`,
  60 additions + 14 deletions = 74 authored lines; assessment passive,
  `review_due=false`. It is complete and accounted separately from implementation.
- Remote delivery remains human-owned and requires separate authorization;
  no remote delivery or merges are authorized here.

### T3 — root-weighted minimum

- Same feature and writer; current branch `feat/dt-weighted-min-gain`, starting
  boundary `f9f89a9376f8496adb2f3d3e85b3738983e5667f`. Preparation began clean;
  implementation resumed with only the inherited tracker preparation modified.
- Implemented signature: `decision_tree(D, A, min_gain=1e-12, max_depth=None,
  min_samples_split=2, depth=0, n_total=None)`. The total initializes once when
  absent, then passes unchanged through every existing recursive call.
- After local score selection, `weighted_gain = len(D) / n_total * (p0 - pg)`
  controls the strict `< min_gain` leaf guard. Root factor is one; descendants use
  original N. Neither ranking assignment is weighted, and Gain Ratio never
  replaces Information Gain in the minimum.
- Existing multibranch conditional entropy supplies `pg`; all rows have unit
  weight. No `sample_weight` support or other model expansion is included.

#### Documentation contract and source references

- Primary reference, scikit-learn 1.9.1 RandomForestClassifier API:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  Its `min_impurity_decrease` formula uses `N_t / N` times the local reduction,
  with N the total training count; equality permits a split. Counts are weighted
  only if sample weights are supplied. The parent checked this source and this
  preparation independently read the same official documentation.
- This aligns threshold weighting only. Custom entropy/multibranch splitting
  still differs from RF's current gini criterion, bootstrap, encoding, and split
  topology. No sklearn runtime comparison, RF execution, or RF change was made.
- At `f9f89a9`, notebook references: `impurityEval_2` at lines 156–161; old local
  minimum explanation at 216–230; signature at 275–276; local ranking at 288–297;
  minimum guard at 298–299; recursive forwarding at 306–311. Changes are confined
  to signature/bookkeeping/guard/docs; the assignment pair is unchanged.
- At that boundary, DT test lines 149–182 constrained the six-argument signature
  and recursion; lines 238–250 required a local-only descendant split. The fixture
  is retained in a shared helper and its regression now asserts weighted behavior.

#### Tests and preservation

- Both in-memory criterion variants run the shared suite; exact two-line
  switching is preserved without writing the notebook during tests.
- Root equivalence is tested with default/explicit original total and gain
  equality/above-threshold cases; existing raw IG versus Gain Ratio cases remain.
- The retained 100-row fixture proves its 10-row child's local gain is 1 and
  root-weighted gain is 0.1. At minimum 0.15 it returns its own majority leaf;
  at equality 0.1 it splits. Both cases retain root-split and majority-label proofs.
- A 15-call trace of three-bit parity recursion proves original N=8 reaches all
  4-row children, 2-row grandchildren, and 1-row leaves. Separate training calls
  with 8 and 2 rows reinitialize their own total instead of sharing mutable state.
- Runtime tests retain all six positional argument meanings; AST assertions
  enforce the seventh default and unchanged `n_total` recursive forwarding.
- All prior depth/sample, categorical, tie, prediction, nonmutation, and default
  output proofs remain; structural/default-output checks below confirm preservation.

#### Baseline, forecast, and handoff

- Observed preparation baseline, run once in the foreground:
  `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_decision_tree -v`:
  PASS, 54 tests in 0.051s on the pre-T3 local-only implementation. That baseline
  was not rerun after interruption; it is not the final weighted-code result.
- Final `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_decision_tree -v`:
  PASS, 62 tests in 0.058s under both criterion variants; no skipped DT checks.
- Final `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -v`:
  63 reported tests in 0.059s, 62 passed, only the known baseline import error
  `unittest.loader._FailedTest.test_random_forest` from missing `sklearn` at
  `test_random_forest.py:12`. No RF notebook execution or installation attempted.
- `git diff --check`: PASS, no reported whitespace errors.
- `git diff --exit-code f9f89a9 -- random-forest.ipynb test_random_forest.py requirements.txt`:
  PASS, no differences.
- Preparation authored forecast: 280–380 additions plus deletions relative to
  `f9f89a9`, including inherited preparation/reconciliation and final evidence: notebook
  35–55, tests 90–135, tracker 155–190. Exclude saved-output churn, not tests or
  explanatory lines. One coherent behavior/tests/docs slice; 400 lines is advisory.
- Read-only comparison against `f9f89a9`: PASS, 11 cells/five code cells; all IDs,
  metadata, execution counts, and saved outputs unchanged. Only `algorithm-description`
  and `c45-algorithm` source changed; local ranking/pair, first-six defaults and
  majority/prediction/display helpers are unchanged. In-memory execution confirms
  complete default example stdout matches the base under both criteria.
- Source-mutating normalization preceded final checks. No formatter was installed
  or run, no unrelated JSON reformatting, saved-output churn, or bytecode writes.
  Runtime boundary: all five notebook code cells execute in the focused test harness.
- Rollback boundary for T3: revert only weighted-minimum notebook/test/docs
  changes relative to `f9f89a9`, preserving T1/T2 controls, reviews, RF, dependencies,
  and the now-completed closeout accounting.
- Parent completed preparation readbacks and authorized source work; next parent
  verifies and commits T3. The checkbox, commit, and separate native assessment/consent
  remain pending. This writer claims no T3 approval, receipt, or delivery;
  no branch change, staging, commit, review, delegation, or remote work was performed.
