# Notebook testing strategy

Objective: recover and complete the existing UCI Heart Disease five-class DT/RF experiment without altering the team algorithm.

Authorized scope: notebook-pruebas.ipynb, its builder and tests, execution artifacts and this recovery document. Preserve previous untracked files; no remote mutations.

Route: delegated direct; preparation and multiple non-trivial files trigger delegation.
TDD: unknown (no explicit project configuration found); use ordinary functional verification, do not claim RED/GREEN evidence.
Delivery: ask-on-risk. Corrections forecast 100–180 authored lines; recording all prior untracked source exceeds 400 lines. Commit and slice strategy pending parent/user decision. No push or PR authorized.
Engram mirror: pending; tools unavailable.

Initial state: feat/NotebookPruebas, main and cached origin/main at f27359e. Notebook/builder/tests untracked. Remote main query failed with getaddrinfo thread failure; actual remote remains unverified.

## Tasks
- [x] T1 Preserve sources and correct provenance, final-result cache and regression checks.
- [x] T2 Execute synthetic verification and real-data experiment if dependencies/network permit; save observed evidence.
- [x] T3 Validate notebook structure and Colab portability; report results and limitations.

Checks: .venv\Scripts\python.exe -B -m unittest test_notebook_pruebas -v; compile all code cells; nbformat validation if available; real 40-configuration/200-fold run and two final fits if dataset available.

Rollback: restore three source files from tmp/notebook-pruebas-backup; generated execution artifacts are isolated.
Commit evidence: pending; no commits made.
Next: parent review/commit and delivery-strategy decision; hosted Colab execution remains unverified.

## Observed implementation and verification

T1: prior builder/notebook/tests preserved in tmp/notebook-pruebas-backup. Dataset fingerprint, original UCI IDs and CSV split/fold assignments added. Validation-input fingerprint binds actual development/reserve X/y and folds; final control rejects changed split/data or cached result from another run/selection. Regression tests cover changed seed/split, configuration and output directory. Team DT sources remain exact.

Command: `.venv\Scripts\python.exe -B -m unittest test_notebook_pruebas -v`: 7/7 passed after final changes (2.686 seconds). Checks include all code-cell compilation, exact grids, original DT snapshot, leakage, selection, provenance and final-cache control.

Dependency installation: sandbox pip failed DNS. Authorized escalated install succeeded for matplotlib 3.11.2, nbformat 5.11.1, nbclient 0.11.0 and dependencies in .venv.

Remote: escalated `git ls-remote origin refs/heads/main` succeeded: ff9469c9df58abae89bce321cac08aae64e8938b. Read-only exact-commit notebook inspection: DT code cells identical to local; remote RF now uses UCI45/dropna/stratified seed55 and RF100 with default Gini. Local refs unchanged; experiment entropy follows accepted plan.

T2: fresh local kernel executed all 40 configurations / 200 CV fits plus 2 final fits against actual UCI45. 297 complete records, development237/reserve60. Original UCI class totals: 160/54/35/35/13. Development:128/43/28/28/10. Reserve:32/11/7/7/3.

Execution history retained:
- First run before split-fingerprint correction: succeeded in 56.593 seconds, folder output/jupyter-notebook/resultados/20261109_214013_295885.
- First corrected-source run: failed after32.404 seconds with OSError22 opening predicciones_parciales.csv, partial folder output/jupyter-notebook/resultados/20261109_214138_548119 (23 configurations). No cause established.
- One fresh corrected-source retry: succeeded in43.025 seconds. Authoritative final results: output/jupyter-notebook/resultados/20261109_214222_541799. 200 detail rows,40 configurations, no notebook error outputs. No metrics reused from partial run.

Selected DT: min_gain1e-12/max_depthNone/min_samples_split25. CV F1macro0.264015±0.015769, CV accuracy0.536082. Final accuracy0.65, F1macro0.352581,25 nodes,depth5,training0.097295 seconds.
Selected RF: entropy/100 trees/max_depth10/min_samples_split2/min_impurity_decrease0. CV F1macro0.285806±0.050690, CV accuracy0.578280. Final accuracy0.616667,F1macro0.317144,13796 total nodes,mean depth9.89,training0.196207 seconds.

T3: nbformat validation passed before/after clean-kernel execution. Source defaults remain disabled. Executed notebook: output/jupyter-notebook/notebook-pruebas-ejecutado.ipynb; includes observed conclusion cell and three embedded PNGs from that run. PNGs visually inspected (final confusion matrices readable); CSVs exported. Self-contained Colab dependency bootstrap, no local-only algorithm imports or hardcoded system paths. Hosted Google Colab not executed. Notebook rerun downloads UCI and requires internet.

Limits: same seed55 reserve previously exposed in original DT outputs; class4 only3 final cases and both models recall0. DT class2 recall0. CV deviations are not confidence intervals; one split/five folds cannot establish statistical superiority. Final ordering reverses CV ordering. Gain Ratio/pruning remain pending. Times measure fit, environment-dependent.

Task checkboxes record observed outcomes; work-unit commits remain pending, not completed. Engram mirror remains pending/unavailable.
