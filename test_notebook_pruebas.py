"""Check the experiment notebook with synthetic data, without running UCI trials."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import ucimlrepo
from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebook-pruebas.ipynb"
TREE_SOURCE = ROOT / "decision-tree.ipynb"
CORE_CELLS = {
    "importaciones", "datos-funciones", "snapshot-node-structure",
    "snapshot-impurity-functions", "snapshot-training-helpers", "snapshot-c45-algorithm",
    "procedencia", "plan-funciones", "evaluacion-funciones",
}


def read_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def namespace_from_notebook():
    notebook = read_notebook(NOTEBOOK)
    namespace = {"sys": sys}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and cell["id"] in CORE_CELLS:
                exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    return namespace


def synthetic_data(n=100):
    """A unique row id lets tests detect accidental final-set training."""
    labels = np.tile(np.arange(5), n // 5)
    return pd.DataFrame({"row_id": np.arange(n), "signal": labels,
                         "noise": np.arange(n) % 7, "target": labels})


class ExperimentNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = read_notebook(NOTEBOOK)
        cls.ns = namespace_from_notebook()

    def test_all_cells_compile_and_default_execution_is_disabled(self):
        cells = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]
                 if cell["cell_type"] == "code"}
        for name, source in cells.items():
            with self.subTest(cell=name):
                compile(source, name, "exec")
        self.assertIn("EJECUTAR_PRUEBAS = False", cells["ejecucion"])
        self.assertIn("EVALUAR_PRUEBA_FINAL = False", cells["final-control"])
        self.assertTrue(all(not cell.get("outputs") for cell in self.notebook["cells"]
                            if cell["cell_type"] == "code"))

    def test_tree_code_is_an_exact_snapshot_of_the_project_algorithm(self):
        original = {cell.get("id"): "".join(cell["source"])
                    for cell in read_notebook(TREE_SOURCE)["cells"]}
        current = {cell.get("id"): "".join(cell["source"])
                   for cell in self.notebook["cells"]}
        for name in ["node-structure", "impurity-functions", "training-helpers", "c45-algorithm"]:
            with self.subTest(cell=name):
                self.assertEqual(current["snapshot-" + name], original[name])

    def test_preparation_keeps_five_classes_and_discards_missing_rows(self):
        raw = synthetic_data()
        raw.loc[0, "noise"] = np.nan
        raw.loc[8, "signal"] = np.nan
        provider_data = SimpleNamespace(data=SimpleNamespace(
            features=raw.drop(columns="target"),
            targets=raw[["target"]].rename(columns={"target": "num"}),
        ))
        with patch.object(ucimlrepo, "fetch_ucirepo", return_value=provider_data) as fetch:
            cleaned = self.ns["cargar_heart_disease"]()
        fetch.assert_called_once_with(id=45)
        self.assertEqual(len(cleaned), 98)
        self.assertEqual(set(cleaned.target), set(range(5)))
        self.assertEqual(cleaned.columns[-1], "target")
        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(cleaned.attrs["ids_uci"], [i for i in range(100) if i not in (0, 8)])
        digest = self.ns["huella_datos"](cleaned)
        self.assertEqual(digest, self.ns["huella_datos"](cleaned.copy()))
        changed = cleaned.copy()
        changed.iloc[0, 0] += 1
        self.assertNotEqual(digest, self.ns["huella_datos"](changed))

    def test_plan_has_agreed_values_and_reuses_identical_configurations(self):
        plan, catalog = self.ns["construir_plan"]()
        self.assertEqual(len(catalog), 40)
        self.assertEqual(len(plan), 51)
        for model in ("DT", "RF"):
            reference = plan[(plan.prueba == "Referencia") & (plan.modelo == model)].iloc[0]
            self.assertEqual(catalog[reference.config_id]["parametros"], self.ns[f"BASE_{model}"])
            depth_none = plan[(plan.prueba == "Profundidad máxima") & (plan.modelo == model)
                              & (plan.valor == "sin límite")].iloc[0]
            self.assertEqual(reference.config_id, depth_none.config_id)
        reference_rf = plan[(plan.prueba == "Referencia") & (plan.modelo == "RF")].iloc[0]
        threshold_rf = plan[(plan.prueba == "Ganancia mínima") & (plan.modelo == "RF")
                            & (plan.valor == "0.0")].iloc[0]
        self.assertEqual(reference_rf.config_id, threshold_rf.config_id)
        self.assertEqual(catalog[reference_rf.config_id]["parametros"]["criterion"], "entropy")
        self.assertFalse(plan.prueba.str.contains("poda|ratio", case=False).any())
        for model in ("DT", "RF"):
            def values(trial, parameter):
                rows = plan[(plan.prueba == trial) & (plan.modelo == model)]
                return {catalog[key]["parametros"][parameter] for key in rows.config_id}
            gain = "min_gain" if model == "DT" else "min_impurity_decrease"
            expected_gain = {0, .001, .005, .01, .05}
            if model == "DT":
                expected_gain.add(1e-12)
            self.assertEqual(values("Ganancia mínima", gain), expected_gain)
            self.assertEqual(values("Profundidad máxima", "max_depth"), {1, 2, 4, 6, 10, None})
            self.assertEqual(values("Mínimo para dividir", "min_samples_split"), {2, 10, 25, 50, 100})
            combos = plan[(plan.prueba == "Profundidad + mínimo") & (plan.modelo == model)]
            self.assertEqual({(catalog[key]["parametros"]["max_depth"],
                               catalog[key]["parametros"]["min_samples_split"])
                              for key in combos.config_id},
                             {(depth, minimum) for depth in (4, 6) for minimum in (2, 10, 50)})
        self.assertEqual(values("Cantidad de árboles", "n_estimators"), {10, 50, 100, 200})

    def test_cross_validation_and_final_evaluation_do_not_train_on_reserved_rows(self):
        data = synthetic_data()
        X_dev, X_final, y_dev, y_final, folds = self.ns["preparar_particiones"](data)
        self.assertEqual((len(X_dev), len(X_final), len(folds)), (80, 20, 5))
        dev_ids, final_ids = set(X_dev.row_id), set(X_final.row_id)
        self.assertFalse(dev_ids & final_ids)

        configs = {
            "dt-test": {"modelo": "DT", "parametros": {**self.ns["BASE_DT"], "max_depth": 2}},
            "rf-test": {"modelo": "RF", "parametros": {**self.ns["BASE_RF"],
                                                           "max_depth": 2, "n_estimators": 4}},
        }
        original_train = self.ns["entrenar_modelo"]
        train_calls = []

        def recording_train(name, X, y, params):
            train_calls.append((name, set(np.asarray(X)[:, 0])))
            return original_train(name, X, y, params)

        with patch.dict(self.ns, {"entrenar_modelo": recording_train}):
            detail, predictions, summary = self.ns["ejecutar_experimentos"](
                X_dev, y_dev, folds, configs, progreso=False)
            self.assertEqual(len(detail), 10)
            self.assertEqual(len(train_calls), 10)
            for _, ids in train_calls:
                self.assertTrue(ids <= dev_ids)
                self.assertFalse(ids & final_ids)

            for model in ("DT", "RF"):
                model_pred = predictions[predictions.modelo == model]
                self.assertEqual(len(model_pred), len(X_dev))
                self.assertEqual(set(model_pred.indice_desarrollo), set(range(len(X_dev))))
                self.assertFalse(model_pred.indice_desarrollo.duplicated().any())
                for fold, rows in model_pred.groupby("particion"):
                    record = detail[(detail.modelo == model) & (detail.particion == fold)].iloc[0]
                    self.assertAlmostEqual(record.accuracy_validacion,
                                           accuracy_score(rows.clase_real, rows.clase_predicha))
                    self.assertAlmostEqual(record.f1_macro_validacion,
                                           f1_score(rows.clase_real, rows.clase_predicha,
                                                    labels=list(range(5)), average="macro", zero_division=0))
                    recalls = recall_score(rows.clase_real, rows.clase_predicha,
                                           labels=list(range(5)), average=None, zero_division=0)
                    for label, value in enumerate(recalls):
                        self.assertAlmostEqual(record[f"recall_clase_{label}"], value)
                summary_row = summary[summary.modelo == model].iloc[0]
                self.assertAlmostEqual(summary_row.f1_macro_validacion_media,
                                       detail[detail.modelo == model].f1_macro_validacion.mean())

            for fold in range(1, 6):
                dt_rows = set(predictions[(predictions.modelo == "DT") &
                                          (predictions.particion == fold)].indice_desarrollo)
                rf_rows = set(predictions[(predictions.modelo == "RF") &
                                          (predictions.particion == fold)].indice_desarrollo)
                self.assertEqual(dt_rows, rf_rows)

            selected = self.ns["seleccionar_configuraciones"](summary)
            final, final_predictions = self.ns["evaluar_reserva"](
                selected, configs, X_dev, y_dev, X_final, y_final)
            self.assertEqual(len(train_calls), 12)
            for _, ids in train_calls[-2:]:
                self.assertEqual(ids, dev_ids)
                self.assertFalse(ids & final_ids)
            for model in ("DT", "RF"):
                rows = final_predictions[final_predictions.modelo == model]
                self.assertEqual(len(rows), len(X_final))
                result = final[final.modelo == model].iloc[0]
                self.assertAlmostEqual(result.accuracy_final,
                                       accuracy_score(rows.clase_real, rows.clase_predicha))
                self.assertAlmostEqual(result.f1_macro_final,
                                       f1_score(rows.clase_real, rows.clase_predicha,
                                                labels=list(range(5)), average="macro", zero_division=0))

    def test_selection_prefers_simpler_candidate_only_within_one_point(self):
        summary = pd.DataFrame([
            {"modelo": "DT", "config_id": "best", "f1_macro_validacion_media": .80,
             "nodos_totales_media": 100},
            {"modelo": "DT", "config_id": "near", "f1_macro_validacion_media": .795,
             "nodos_totales_media": 20},
            {"modelo": "DT", "config_id": "far", "f1_macro_validacion_media": .78,
             "nodos_totales_media": 2},
            {"modelo": "RF", "config_id": "rf-best", "f1_macro_validacion_media": .81,
             "nodos_totales_media": 1000},
        ])
        chosen = self.ns["seleccionar_configuraciones"](summary).set_index("modelo")
        self.assertEqual(chosen.loc["DT", "config_id"], "near")
        self.assertEqual(chosen.loc["RF", "config_id"], "rf-best")

    def test_notebook_workflow_exports_results_and_does_not_repeat_final_fit(self):
        # Exercise the notebook's cells and file outputs, using fake plots and data.
        class Axis:
            def errorbar(self, *args, **kwargs): pass
            def set(self, *args, **kwargs): pass
            def legend(self, *args, **kwargs): pass
            def imshow(self, *args, **kwargs): pass
            def text(self, *args, **kwargs): pass

        class Figure:
            def tight_layout(self): pass
            def savefig(self, path, **kwargs): Path(path).write_bytes(b"plot-placeholder")

        def subplots(rows, columns, figsize=None, squeeze=True):
            axes = np.array([[Axis() for _ in range(columns)] for _ in range(rows)], dtype=object)
            return Figure(), axes.squeeze(axis=0) if squeeze and rows == 1 else axes

        pyplot = types.ModuleType("matplotlib.pyplot")
        pyplot.subplots, pyplot.show = subplots, lambda: None
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        data = synthetic_data()
        fake = SimpleNamespace(data=SimpleNamespace(
            features=data.drop(columns="target"),
            targets=data[["target"]].rename(columns={"target": "num"}),
        ))
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, \
             patch.object(ucimlrepo, "fetch_ucirepo", return_value=fake), \
             patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}), \
             contextlib.redirect_stdout(io.StringIO()):
            namespace = {"sys": sys}
            final_source = None
            for cell in self.notebook["cells"]:
                if cell["cell_type"] != "code" or cell["id"] == "dependencias":
                    continue
                source = "".join(cell["source"])
                if cell["id"] == "ejecucion":
                    source = source.replace("EJECUTAR_PRUEBAS = False", "EJECUTAR_PRUEBAS = True")
                    source = source.replace('Path("resultados_notebook_pruebas")', f"Path({temp!r})")
                if cell["id"] == "final-control":
                    source = source.replace("EVALUAR_PRUEBA_FINAL = False", "EVALUAR_PRUEBA_FINAL = True")
                    final_source = source
                exec(compile(source, cell["id"], "exec"), namespace)
                if cell["id"] == "plan-listado":
                    plan = namespace["plan"]
                    chosen = plan[(plan.prueba == "Referencia") |
                                  ((plan.prueba == "Profundidad máxima") & (plan.valor == "4")) |
                                  ((plan.prueba == "Cantidad de árboles") & (plan.valor == "10"))]
                    keys = set(chosen.config_id)
                    namespace["plan"] = plan[plan.config_id.isin(keys)].copy()
                    namespace["catalogo"] = {key: config for key, config in namespace["catalogo"].items()
                                               if key in keys}
            output = namespace["CARPETA_RESULTADOS"]
            expected = ["asignaciones_reserva.csv", "asignaciones_folds.csv", "plan_de_pruebas.csv", "metadatos.json", "resultados_parciales.csv",
                        "predicciones_parciales.csv", "configuraciones_parciales.csv",
                        "resultados_por_particion.csv", "resultados_por_configuracion.csv",
                        "resultados_por_prueba.csv", "predicciones_validacion.csv",
                        "configuraciones_seleccionadas.csv", "graficos_validacion.png",
                        "matrices_validacion.png", "resultados_prueba_final.csv",
                        "predicciones_prueba_final.csv", "matrices_prueba_final.png"]
            for filename in expected:
                with self.subTest(output=filename):
                    self.assertTrue((output / filename).is_file())
            self.assertEqual(len(pd.read_csv(output / "resultados_prueba_final.csv")), 2)
            self.assertEqual(len(pd.read_csv(output / "resultados_por_particion.csv")), 25)
            self.assertEqual(json.loads((output / "metadatos.json").read_text(encoding="utf-8"))["clases"],
                             list(range(5)))

            assignments = pd.read_csv(output / "asignaciones_reserva.csv")
            self.assertEqual(len(assignments), len(data))
            self.assertEqual(len(assignments.id_uci.unique()), len(data))
            fold_assignments = pd.read_csv(output / "asignaciones_folds.csv")
            self.assertEqual(len(fold_assignments), 5 * 80)
            self.assertEqual(set(fold_assignments[fold_assignments.rol == "validacion"].id_uci),
                             set(assignments[assignments.conjunto == "desarrollo"].id_uci))

            def unexpected_fit(*args, **kwargs):
                raise AssertionError("The final cell must not retrain on a second execution")
            namespace["entrenar_modelo"] = unexpected_fit
            exec(compile(final_source, "final-control-again", "exec"), namespace)
            saved_seed = namespace["SEMILLA_RESERVA"]
            namespace["SEMILLA_RESERVA"] = saved_seed + 1
            changed = namespace["preparar_particiones"](namespace["datos"])
            names = ["X_desarrollo", "X_final", "y_desarrollo", "y_final", "particiones"]
            saved = {name: namespace[name] for name in names}
            namespace.update(dict(zip(names, changed)))
            with self.assertRaisesRegex(RuntimeError, "particiones cambiaron"):
                exec(compile(final_source, "final-control-changed-split", "exec"), namespace)
            namespace.update(saved)
            namespace["SEMILLA_RESERVA"] = saved_seed
            key = namespace["elegidas"].iloc[0]["config_id"]
            namespace["catalogo"][key]["parametros"]["max_depth"] = 999
            with self.assertRaisesRegex(RuntimeError, "otra corrida o selección"):
                exec(compile(final_source, "final-control-changed-config", "exec"), namespace)
            namespace["CARPETA_RESULTADOS"] = Path(temp) / "another-run"
            with self.assertRaisesRegex(RuntimeError, "otra corrida o selección"):
                exec(compile(final_source, "final-control-stale", "exec"), namespace)
            self.assertFalse(namespace["CARPETA_RESULTADOS"].exists())


if __name__ == "__main__":
    unittest.main()
