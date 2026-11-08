# Extend the existing tree with continuous splits

Let numeric attributes compete with categorical attributes in the existing recursive trainer, using the best information-gain threshold at each node and allowing continuous reuse.

## Scope and decisions

- User selected automatic numeric-column detection and TDD for this feature; no new training parameters.
- Numeric feature columns are continuous; other feature columns are categorical. Class labels remain categorical. Boolean features remain categories, not measurements.
- For continuous attributes, sort distinct observed values and evaluate every consecutive midpoint. Choose the first best information-gain cut, then compare with categorical candidates.
- Reuse a selected continuous attribute; remove a selected categorical attribute. Skip continuous attributes without a valid two-sided split; if no candidate exists, return a supported majority leaf.
- Preserve the current signature/defaults, visible pseudocode flow, depth/sample limits, root-weighted raw-gain minimum, stable attribute ties, class ties, leaf support, and categorical unseen-value fallback.
- Information Gain remains active. Preserve the existing optional GR switch as an explicit alternative; continuous threshold search itself maximizes raw gain.
- Preserve helpers / algorithm-only / drawing / execution cells. Update numeric prediction and interval labels consistently.
- Assume well-formed nonempty input with finite comparable numeric measurements and no missing values; no validation framework, imputation, pruning, weighting, new trainer, or dependency installation.
- Authorized files: `decision-tree.ipynb`, `test_decision_tree.py`, and this tracker. Preserve existing dirty `test_decision_tree_training.py` exactly; do not modify alternative notebooks or frozen historical tests.
- No staging, commits, push, PR, installs, or remote operations without explicit authorization.

## Work unit

- [ ] **T1 — Extend continuous split support in place.** Route: delegated direct; trigger: coordinated notebook/test implementation and write preparation.
  - Record current baseline and snapshots before edits.
  - Add tests for detection, all candidate cuts, best gain, mixed competition, continuous reuse, constant-column termination, prediction boundaries, and labels. Observe RED before production edits.
  - Add only cohesive threshold/type helpers, optional node/helper threshold state, and minimal changes to the existing attribute loop/partition/recursion.
  - Preserve original semantic regression coverage; update numeric-as-category expectations because the user explicitly chose automatic numeric inference.
  - Verify GREEN, refactor, full suite, output preservation, and protected files.
  - Record actual evidence, limitations, rollback, and assessment; commit pending explicit authorization.

## Checks and delivery

- TDD: **on**, explicit user selection for this feature. Runner: Python `unittest`; RED, GREEN, REFACTOR.
- Focused/runtime: `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -B -m unittest test_decision_tree -v`.
- Full: `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -B -m unittest discover -v`.
- Structural: `git diff --check`, ordered cells, unchanged datasets/evaluation, protected test/alternative files, unchanged saved outputs/metadata.
- Known historical full-suite failures: alternate notebook bindings/cell/heading and missing `ipywidgets`/`sklearn`; real matplotlib rendering skipped. Reconfirm current baseline.
- RDD: read user-owned mode and native assessment after normalization; prior candidate declines do not apply to this new change.
- Delivery: `ask-on-risk`; preliminary forecast 300–500 authored changed lines, subject to actual implementation. Approximately 400 per work unit is advisory, not a reason to compress code or omit tests. No commit or PR planned.
- Branch: retain current `feat/categorical-only-decision-tree`; running committed authored lines: 0 for this feature.

## Progress and recovery

- Requested behavior implemented; focused GREEN, full verification partial due to unchanged baseline failures. T1 remains unchecked; parent native assessment is pending and was not run here.
- Repository-relative locator: `odd/tasks/decision-tree-continuous-attributes.md`.
- Mirror: `odd/decision-tree-continuous-attributes/tasks`.
- Rollback boundary: only this feature's numeric type/cut/helper/node/prediction/display changes, direct regression updates, and this tracker; preserve pre-existing edits.
- Next step: parent assessment of this normalized local candidate; no delivery is authorized.

## Implementation evidence

- Snapshot: current HEAD `0cce0eea95c7a29b54ecb2d7f8c2a72eeaf129b8`, not an earlier assumed revision. Notebook and main test were clean; only the training test was dirty before this new tracker. Initial bytes/hashes and tracked blob identities were captured before edits.
- Added `is_continuous` (4 source lines) and `best_threshold` (12); main trainer is 51 source lines. Sizes are informational, not acceptance gates.
- Main retains pure / empty-attributes / depth / sample checks, `else` impurity calculation, actual per-attribute loop and conditional impurity, selection, strict minimum `if/else`, node creation, partition, nonempty branches, and direct recursion. Signature/defaults are unchanged.
- Numeric candidates use sorted distinct adjacent half-sum midpoints, first-best raw IG, and a lower-bound rounding fallback. Selected continuous attributes remain available; categorical attributes are removed. No numeric candidate returns a supported majority leaf even at zero minimum.
- `Node`, `partition`, `impurityEval_2`, and `split_information` accept optional threshold state. Existing two-argument categorical helper calls retain observed-value grouping. Numeric prediction, text, layout, and mock plotting use consistent `<=` / `>` intervals.
- Numeric inference uses current feature values (`int`/`float`, excluding `bool`); numeric codes are continuous and numeric targets remain classes. Assume finite comparable ordinary measurements, no missing values, and no validation framework; arbitrary-size integer arithmetic beyond float conversion is not promised.
- Raw IG is active throughout. The adjacent optional GR switch ranks attributes after raw-IG cut search; the minimum remains root-weighted raw IG. Contradictory notebook explanations were updated without altering untouched Spanish comments.

### TDD and final commands

Commands were run in the foreground with no installs. Final runs followed targeted JSON-indentation/test-alignment normalization, not whole-notebook serialization.

| Stage | Exact command | Observed result |
|---|---|---|
| Baseline focus | `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -B -m unittest test_decision_tree -v` | 96 tests; 94 passed, 2 skipped |
| Baseline full | `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -B -m unittest discover -v` | 140 tests; 1 failure, 10 errors, 2 skipped |
| RED, before notebook edits | `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -B -m unittest test_decision_tree -v` | 104 tests; 17 failures, 22 errors, 2 skipped (includes subtests) |
| Intermediate | Same focused command | 104 tests; 2 incorrect repeated-value expectations failed; hand calculations showed gains ~0.0817 at 0.75 and ~0.1909 at 5.75, so expectations were corrected |
| GREEN | Same focused command | 108 tests; 106 passed, 2 skipped |
| Final focus | `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -B -m unittest test_decision_tree -v` | 110 tests; 108 passed, 2 skipped |
| Final full | `PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg python3 -B -m unittest discover -v` | 154 tests; 141 passed, same 1 failure and 10 errors, 2 skipped; NOT full-suite PASS |
| Whitespace | `git diff --check` | Passed |

Both criterion suites skip `test_plot_artists_and_same_cell_display` because matplotlib is absent. Mock plot checks pass; real rendering is not verified.

Unchanged full-suite failure: `test_decision_tree_training.TrainingReviewNotebookTests.test_markdown_is_spanish` (alternate notebook heading mismatch).

Unchanged full-suite errors:
- `test_decision_tree_training.EnrichedTrainingTests.test_enriched_execution_preserves_in_memory_baseline_and_helpers` and `test_quinlan_leaf_replacement_is_real_and_original_is_unchanged`: missing `tree`.
- `test_decision_tree_training.EnrichedTrainingTests.test_numeric_reuse_whole_training_reference_and_no_mutation` and `test_public_api_matches_pre_refactor_behavioral_snapshot`: missing `print_tree`.
- `test_decision_tree_training.TrainingReviewNotebookTests.test_original_training_is_copied_without_integration`: missing alternate `c45-algorithm` cell.
- `test_enriched_tree_panel.EnrichedTreePanelTests.test_click_trains_and_prints_without_replacing_original_models_or_data`, `test_invalid_controls_do_not_train_or_replace_the_last_result`, `test_panel_defaults_are_lazy_and_match_the_trainer`, and `test_selected_parameters_and_weights_reach_the_existing_public_trainer`: missing `ipywidgets`.
- `unittest.loader._FailedTest.test_random_forest`: missing `sklearn`.

### Preservation and recovery

- Differential execution against the immutable pre-edit snapshot matched six categorical controls under each of IG and GR: example, raw-gain tradeoff, descendant root weighting, Boolean parity, numeric class labels, and zero-gain constant category. Compared recursive state/support, predictions, text tree, and notebook stdout. Numeric fixtures intentionally changed semantics and were validated independently.
- Compared all notebook non-source fields, cell order/IDs, metadata, execution counts, saved PNG/outputs, and complete training-data/training-execution/test-accuracy cells against that snapshot: unchanged. No notebook execution was persisted.
- Initial SHA256: notebook `512c91b27f5ef25d8b443147ae68dbd579db7cae56fca08e4b530fafc6ef991d`; main test `2149062f06ecd78ac735c8a2dbc519deb40be69caa51a32f20de15f61ec7c617`; tracker `e4c92970e8748d99773c19154066b712bb86f64a0d7fa9a27da5d5fdf677eeb4`.
- Protected dirty `test_decision_tree_training.py` SHA256 before/after: `805622bb40054ac7f9e235c437133b14a54aea668a5f52b84eca494ae279d0d8`. Its existing 18 additions / 3 deletions are not part of this work. All other tracked files retain initial index blob identities and no new differences.
- Rollback: reverse only this feature's notebook/main-test hunks and restore this tracker's initial contents; do not reset the repository or touch the dirty training test. No rollback was performed.
- Authored notebook/test diff: 117 + 201 = 318 changed lines; complete new tracker: 92 lines; feature-wide total: 410. This writer changed 367 lines relative to the supplied three-file snapshot (318 plus 47 tracker additions / 2 removals). The small advisory overage preserves explicit regression/preservation evidence rather than compressing code. No slicing or delivery actions are authorized.
- Skill resolution: `paths-injected`; read exact `/Users/valentinohonnorat/.config/opencode/skills/work-unit-commits/SKILL.md` and `/Users/valentinohonnorat/.config/opencode/skills/cognitive-doc-design/SKILL.md` before task work.
- Mirror target: project `decision-tree-research`, topic `odd/decision-tree-continuous-attributes/tasks`, with this full document and relative locator. Memory persistence/read-back is reported separately; it is not verification or native review.
- Parent spot check: 110 focused tests, OK, skipped=2; `git diff --check` passed. Independent read-only verification repeated focused/full results and confirmed protected files, outputs, visible flow, and signatures.
- Independent entropy-oracle checks passed for all three age-example cuts, mixed competition in both directions, continuous reuse, constant-column termination, boundary routing, and 987 nonconstant seeded random cases. No defect was found in the finite, nonmissing measurement scope; real rendering and unrelated baseline failures remain unresolved.
