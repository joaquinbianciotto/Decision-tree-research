"""Comprueba el panel con widgets reales, sin necesitar un frontend Jupyter."""

import contextlib
from copy import deepcopy
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_decision_tree_training import ROOT, execute


class EnrichedTreePanelTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
        self.assertIn("interactive-panel", [cell["id"] for cell in notebook["cells"]])
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.ns, self.stdout, self.stderr = execute(cells, with_original=True)
        self.panel = self.ns["crear_panel_arbol_enriquecido"]()

    def click(self, panel=None):
        panel = self.panel if panel is None else panel
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            panel.boton_entrenar.click()
        self.assertEqual(stderr.getvalue(), "")
        self.assertFalse(panel.boton_entrenar.disabled)
        return stdout.getvalue()

    def state(self, node):
        return (node.prediction, node.attribute, node.threshold, node.class_probabilities,
                {branch: self.state(child) for branch, child in node.children.items()})

    def test_branch_labels_include_attribute_names_and_full_conditions(self):
        for low, high in ((0, 1), ("calmo", "fuerte")):
            with self.subTest(categories=(low, high)):
                rows = ([[50, low, "no"]] * 2 + [[50, high, "no"]] * 2
                        + [[70, low, "sí"]] * 2 + [[70, high, "no"]] * 2)
                tree = self.ns["entrenar_arbol_enriquecido"](
                    rows, [0, 1], continuos=[0], podar=False)
                before = deepcopy(self.state(tree))
                predictions = [self.ns["predecir_enriquecido"](tree, row[:-1]) for row in rows]
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = self.ns["imprimir_arbol_enriquecido"](
                        tree, ["temperatura", "viento", "Clase"])
                self.assertIsNone(result)
                self.assertEqual(stderr.getvalue(), "")
                self.assertEqual(stdout.getvalue(), (
                    "temperatura\n"
                    "  [temperatura <= 50]\n"
                    "    Class: no\n"
                    "  [temperatura > 50]\n"
                    "    viento\n"
                    f"      [viento = {low!r}]\n"
                    "        Class: sí\n"
                    f"      [viento = {high!r}]\n"
                    "        Class: no\n"))
                self.assertEqual(self.state(tree), before)
                self.assertEqual(tree.total_weight, len(rows))
                self.assertEqual(
                    [self.ns["predecir_enriquecido"](tree, row[:-1]) for row in rows], predictions)

    def test_printing_leaf_retains_class_output_and_indentation(self):
        tree = self.ns["entrenar_arbol_enriquecido"]([[50, "no"], [70, "no"]], [0])
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = self.ns["imprimir_arbol_enriquecido"](tree, ["temperatura", "Clase"], "  ")
        self.assertIsNone(result)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("  Class: no\n", ""))
        self.assertEqual(tree.prediction, "no")

    def test_kernel_panel_uses_only_cleveland_with_continuous_attributes_selected(self):
        notebook = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "interactive-panel")
        namespace = dict(self.ns, get_ipython=lambda: SimpleNamespace(kernel=object()))
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("IPython.display.display") as display, contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
        panel = namespace["panel_enriquecido"]
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
        display.assert_called_once_with(panel)
        self.assertEqual(panel.controles["continuos"].value, (0, 3, 4, 7, 9))
        self.assertFalse(panel.controles["min_peso_rama"].disabled)
        output = self.click(panel)
        self.assertNotIn("No se pudo entrenar:", output)
        self.assertEqual(panel.arbol.total_weight, len(namespace["filas_cleveland"]))
        self.assertEqual(set(panel.arbol.class_probabilities), {0, 1, 2, 4})
        self.assertEqual(panel.parametros_utilizados["continuos"], (0, 3, 4, 7, 9))

    def test_panel_defaults_are_lazy_and_match_the_trainer(self):
        import ipywidgets as widgets

        self.assertIsInstance(self.panel, widgets.VBox)
        self.assertIsInstance(self.panel.salida, widgets.Output)
        self.assertIsInstance(self.panel.boton_entrenar, widgets.Button)
        self.assertIsNone(self.panel.arbol)
        controls = self.panel.controles
        self.assertEqual(controls["min_gain"].value, 1e-12)
        self.assertEqual(controls["min_peso_rama"].value, 2.0)
        self.assertEqual(controls["confianza"].value, 0.25)
        self.assertTrue(controls["podar"].value)
        self.assertTrue(controls["sin_limite"].value)
        self.assertTrue(controls["max_depth"].disabled)
        self.assertEqual(controls["min_samples_split"].value, 2)
        self.assertEqual(controls["continuos"].options, tuple(
            (self.ns["columns"][index], index) for index in self.ns["A"]))
        self.assertEqual(controls["continuos"].value, ())
        self.assertFalse(controls["continuos"].disabled)
        self.assertTrue(controls["min_peso_rama"].disabled)
        controls["podar"].value = False
        self.assertTrue(controls["confianza"].disabled)
        controls["sin_limite"].value = False
        self.assertFalse(controls["max_depth"].disabled)
        controls["podar"].value = True
        self.assertFalse(controls["confianza"].disabled)

    def test_click_trains_and_prints_without_replacing_original_models_or_data(self):
        original_tree, original_enriched = self.ns["tree"], self.ns["arbol_enriquecido"]
        before = deepcopy((self.ns["D"], self.ns["A"], self.ns["training_matrix"]))
        self.panel.controles["continuos"].value = tuple(self.ns["continuos_cleveland"])
        output = self.click()
        self.assertEqual(self.state(self.panel.arbol), self.state(original_enriched))
        expected = io.StringIO()
        with contextlib.redirect_stdout(expected):
            print("Árbol enriquecido entrenado:")
            self.ns["imprimir_arbol_enriquecido"](self.panel.arbol, self.ns["columns"])
            print(f"Distribución de clases en la raíz: {self.panel.arbol.class_probabilities}")
        self.assertTrue(output.endswith(expected.getvalue()), repr(output))
        self.assertIs(self.ns["tree"], original_tree)
        self.assertIs(self.ns["arbol_enriquecido"], original_enriched)
        self.assertEqual((self.ns["D"], self.ns["A"], self.ns["training_matrix"]), before)
        controls = self.panel.controles
        controls["sin_limite"].value = False
        controls["max_depth"].value = 0
        self.click()
        self.assertIsNone(self.panel.arbol.attribute)
        controls["sin_limite"].value = True
        controls["min_samples_split"].value = len(self.ns["D"]) + 1
        self.click()
        self.assertIsNone(self.panel.arbol.attribute)

    def test_selected_parameters_and_weights_reach_the_existing_public_trainer(self):
        rows = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"]]
        weights = [1.0, 2.0, 1.0, 2.0]
        before = deepcopy((rows, weights))
        panel = self.ns["crear_panel_arbol_enriquecido"](
            rows, [0], ["Medición", "Clase"], pesos=weights)
        controls = panel.controles
        self.assertEqual(controls["continuos"].options, (("Medición", 0),))
        controls["continuos"].value = (0,)
        self.assertFalse(controls["min_peso_rama"].disabled)
        for name, value in {"min_gain": 0.1, "min_peso_rama": 1.5, "confianza": 0.4,
                            "podar": False, "max_depth": 1, "min_samples_split": 3,
                            "sin_limite": False}.items():
            controls[name].value = value
        trainer = self.ns["entrenar_arbol_enriquecido"]
        spy = Mock(wraps=trainer)
        self.ns["entrenar_arbol_enriquecido"] = spy
        self.click(panel)
        spy.assert_called_once_with(
            rows, [0], continuos=(0,), pesos=weights, min_gain=0.1,
            min_peso_rama=1.5, confianza=0.4, podar=False,
            max_depth=1, min_samples_split=3, criterio="gain", filtro_ganancia_promedio=True,
            min_gain_ratio=None, correccion_mdl=False, min_samples_leaf=0, min_weight_leaf=0.0,
            metodo_poda="ninguna", subtree_raising=True, ccp_alpha=0.0)
        expected = trainer(rows, [0], continuos=[0], pesos=weights, min_gain=0.1,
                           min_peso_rama=1.5, confianza=0.4, podar=False,
                           max_depth=1, min_samples_split=3)
        self.assertEqual(self.state(panel.arbol), self.state(expected))
        self.assertEqual(panel.arbol.attribute, 0)
        self.assertEqual(panel.arbol.threshold, 2)
        self.assertEqual((rows, weights), before)
        controls["min_peso_rama"].value = 3.1
        self.click(panel)
        self.assertIsNone(panel.arbol.attribute)
        controls["continuos"].value = ()
        self.assertTrue(controls["min_peso_rama"].disabled)

    def test_new_controls_are_grouped_and_follow_the_selected_method(self):
        import ipywidgets as widgets

        self.assertEqual(tuple(self.panel.grupos), ("División", "Crecimiento", "Poda", "Resultados"))
        self.assertTrue(all(isinstance(group, widgets.VBox) for group in self.panel.grupos.values()))
        c = self.panel.controles
        for name, value in {"criterio": "gain", "filtro_ganancia_promedio": True,
                            "correccion_mdl": False, "min_samples_leaf": 0,
                            "min_weight_leaf": 0.0, "metodo_poda": "pesimista",
                            "subtree_raising": True, "ccp_alpha": 0.0}.items():
            self.assertEqual(c[name].value, value)
        self.assertTrue(c["filtro_ganancia_promedio"].disabled)
        c["criterio"].value = "gain_ratio"
        self.assertFalse(c["filtro_ganancia_promedio"].disabled)
        c["usar_min_gain_ratio"].value = True
        self.assertFalse(c["min_gain_ratio"].disabled)
        for method in ("cost_complexity", "reduced_error", "ninguna", "pesimista"):
            c["metodo_poda"].value = method
            self.assertEqual(c["ccp_alpha"].disabled, method != "cost_complexity")
            self.assertEqual(c["confianza"].disabled, method != "pesimista")
            self.assertEqual(c["subtree_raising"].disabled, method != "pesimista")
            self.assertEqual(c["filas_validacion"].disabled, method != "reduced_error")
            self.assertEqual(c["pesos_validacion"].disabled, method != "reduced_error")
        c["podar"].value = False
        self.assertTrue(c["metodo_poda"].disabled)
        self.assertTrue(c["confianza"].disabled)

    def test_standalone_panel_prints_structure_without_loading_original_algorithm(self):
        notebook = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
        ns, stdout, stderr = execute([c for c in notebook["cells"] if c["cell_type"] == "code"])
        panel = ns["crear_panel_arbol_enriquecido"]()
        output = self.click(panel)
        self.assertNotIn("decision_tree", ns)
        self.assertNotIn("print_tree", ns)
        self.assertNotIn("tree", ns)
        self.assertEqual((stdout, stderr), ("", ""))
        self.assertIsNotNone(panel.arbol)
        self.assertIn("Antes de podar:", output)
        self.assertIn("Después de podar:", output)
        self.assertIn("Parámetros utilizados:", output)
        self.assertIn("Profundidad", output)
        self.assertIn("Nodos", output)
        self.assertIn("Hojas", output)
        self.assertEqual(panel.resultados["despues"], ns["estadisticas_arbol_enriquecido"](panel.arbol))
        self.assertEqual(panel.parametros_utilizados["criterio"], "gain")

    def test_validation_is_explicit_editable_and_never_reuses_training_data(self):
        rows = [["a", "no"], ["b", "yes"], ["b", "yes"]]
        validation = [["a", "yes"], [None, "yes"]]
        weights = [.25, 2.0]
        before = deepcopy((rows, validation, weights))
        panel = self.ns["crear_panel_arbol_enriquecido"](
            rows, [0], ["X", "Clase"], filas_validacion=validation, pesos_validacion=weights)
        panel.controles["metodo_poda"].value = "reduced_error"
        output = self.click(panel)
        self.assertNotIn("No se pudo entrenar:", output)
        self.assertEqual(panel.parametros_utilizados["filas_validacion"], validation)
        self.assertEqual(panel.parametros_utilizados["pesos_validacion"], weights)
        self.assertEqual((rows, validation, weights), before)
        last = panel.arbol
        panel.controles["filas_validacion"].value = ""
        output = self.click(panel)
        self.assertIn("No se pudo entrenar:", output)
        self.assertIn("validacion", output)
        self.assertIs(panel.arbol, last)
        panel.controles["filas_validacion"].value = "not JSON"
        self.assertIn("No se pudo entrenar:", self.click(panel))
        self.assertIs(panel.arbol, last)

    def test_new_options_and_continuous_mdl_reach_the_trainer(self):
        rows = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"]]
        panel = self.ns["crear_panel_arbol_enriquecido"](rows, [0], ["X", "Clase"])
        values = {"criterio": "gain_ratio", "filtro_ganancia_promedio": False,
                  "usar_min_gain_ratio": True, "min_gain_ratio": .1,
                  "correccion_mdl": True, "continuos": (0,),
                  "min_samples_leaf": 2, "min_weight_leaf": 1.5,
                  "metodo_poda": "cost_complexity", "ccp_alpha": .3}
        for name, value in values.items():
            panel.controles[name].value = value
        spy = Mock(wraps=self.ns["entrenar_arbol_enriquecido"])
        self.ns["entrenar_arbol_enriquecido"] = spy
        self.assertNotIn("No se pudo entrenar:", self.click(panel))
        spy.assert_called_once()
        kwargs = spy.call_args.kwargs
        for name, value in values.items():
            if name != "usar_min_gain_ratio":
                self.assertEqual(kwargs[name], value)
        self.assertNotIn("filas_validacion", kwargs)
        self.assertLessEqual(panel.resultados["despues"]["hojas"], panel.resultados["antes"]["hojas"])

    def test_invalid_controls_do_not_train_or_replace_the_last_result(self):
        rows = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"]]
        panel = self.ns["crear_panel_arbol_enriquecido"](rows, [0], ["X", "Clase"])
        controls = panel.controles
        controls["continuos"].value = (0,)
        controls["sin_limite"].value = False
        self.click(panel)
        previous = panel.arbol
        before = deepcopy(rows)
        spy = Mock(wraps=self.ns["entrenar_arbol_enriquecido"])
        self.ns["entrenar_arbol_enriquecido"] = spy
        for name, value in (("min_gain", -1), ("min_gain", float("nan")),
                            ("min_gain", float("inf")), ("min_peso_rama", 0),
                            ("confianza", 0), ("confianza", 1),
                            ("max_depth", -1), ("min_samples_split", 1),
                            ("min_samples_leaf", -1), ("min_weight_leaf", -1),
                            ("min_weight_leaf", float("nan")), ("ccp_alpha", -1)):
            with self.subTest(parameter=name, value=value):
                original = controls[name].value
                controls[name].value = value
                output = self.click(panel)
                self.assertIn("No se pudo entrenar:", output)
                self.assertIn(name, output)
                self.assertIs(panel.arbol, previous)
                self.assertEqual(rows, before)
                spy.assert_not_called()
                controls[name].value = original


if __name__ == "__main__":
    unittest.main()
