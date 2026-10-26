# Decision Tree criteria and stopping parameters

## Objective and authorized scope

Extend the custom teaching Decision Tree with an easy Information Gain / Gain
Ratio switch, maximum depth, and minimum samples for splitting. Information Gain
remains active; the adjacent Gain Ratio assignment is commented out and uses the
same score variable. Keep the existing minimum raw gain behavior.

- Edit `decision-tree.ipynb`, `test_decision_tree.py`, and this document only.
- Do not modify Random Forest, its tests, requirements, or unrelated work.
- No dependency installation, remote execution, push, pull request, or merge.
- Branch: `feat/dt-criteria-stopping`; starting boundary: `b524d86`.

## Decisions and acceptance criteria

- Normalize Gain Ratio by the entropy of branch proportions; handle zero safely.
- Rank attributes by the active score; keep first-attribute tie behavior.
- Preserve `min_gain=1e-12` as raw node-local Information Gain, including equality.
- Root depth is zero; `max_depth=None` is unlimited; zero creates a majority leaf.
- `min_samples_split=2` by default; fewer samples create a majority leaf.
- Propagate both stopping settings through direct recursion.
- Test both criterion lines by switching copied source in memory, not on disk.
- Preserve notebook cell IDs and saved outputs; avoid unrelated JSON churn.
- Reconcile the stale stdout assertion with existing predictions and accuracy
  output in the DT test harness; do not change runtime output to hide the failure.

## Testing configuration and baseline

- TDD: **off for this feature**, explicitly selected by the user as `Sin TDD`.
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
- [ ] **T2 — Stopping parameters:** add depth and minimum-sample controls,
  recursive propagation, boundary tests, and usage documentation.
  Route: delegated direct; same two-file writer trigger. Commit: pending.

## Delivery and review evidence

- Initial authored forecast: T1 250–340 lines; T2 170–240; total 420–580 additions
  plus deletions, including tracking and excluding saved-output churn.
- About 400 lines per task is advisory, not a reason to omit tests or minify.
- Delivery strategy: `ask-on-risk`; user selected `feature-branch-chain`.
- Local tracker: `feat/dt-criteria-stopping`; T1: `feat/dt-criteria` from tracker;
  T2: `feat/dt-stopping-parameters` from T1. No PRs, push, or merges authorized.
- Running committed authored count: 304 (T1: 270 additions, 34 deletions).
- RDD: on, deciding source default. Native assessment/review is parent-owned.
- Last reviewed boundary: `74ea6b6`; T1 assessed high, consent granted, reviewed,
  and acknowledged with authority burned. T2 assessment/consent is independent.
- Rollback: revert each work-unit commit without touching RF or dependencies.

## Progress and next step

T1 is complete on `feat/dt-criteria` at `74ea6b6`; the parent spot check passed
all 40 DT tests. Native four-lens review found no issues and acknowledgement
confirmed authority burned for lineage `review-ecd389460b7f5458`.
T2 source and tests are implemented on `feat/dt-stopping-parameters`; its task
remains unchecked and its commit is pending parent verification and creation.
No T2 review, approval, receipt, or delivery is claimed.

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
- Native committed assessment: high (`process_boundary`), review due; user granted
  this candidate's review. All four lenses reported no findings. Exact native
  acknowledgement returned `action=acknowledged` and `authority=burned`.
- Earlier uncommitted assessment was unassessable due to the untracked tracker;
  it did not lower risk. Committed assessment resolved the inventory normally.
- Proceed to T2; no push, PR, merge, or main change has occurred.

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
- T2 commit is pending. No staging, commits, branch changes, native review,
  delegation, installation, remote operations, or delivery were performed.
- Parent verifies and commits T2, then owns its separate assessment/consent/review.
