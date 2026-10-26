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

- [ ] **T1 — Criterion switch:** add normalized Gain Ratio alongside active
  Information Gain, preserve raw minimum gain, add behavior tests and notebook
  explanations, and reconcile the stale DT stdout expectation.
  Route: delegated direct; notebook + tests are two non-trivial files, and
  preparation belongs to the same bounded writer. Commit: pending.
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
- Running committed authored count: zero; no implementation commits yet.
- RDD: on, deciding source default. Native assessment/review is parent-owned.
- Last reviewed boundary: `b524d86`; assessment and candidate consent pending.
- Rollback: revert each work-unit commit without touching RF or dependencies.

## Progress and next step

T1 source and tests are implemented on `feat/dt-criteria`. T1 remains unchecked
and its commit is pending parent verification and commit creation. T2 has not
started; no depth or sample-count parameters were added. Native review remains
parent-owned and pending, with no approval or receipt claimed.

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
- No staging, commits, branch changes, reviews, remote operations, or delegation.
- Full tracker mirror target: Engram observation `4940`, topic
  `odd/decision-tree-criteria-stopping/tasks`.
- Parent verifies and commits T1 before authorizing T2 on its dependent branch.
