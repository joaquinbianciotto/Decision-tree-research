"""Comprueba el panel con widgets reales, sin necesitar un frontend Jupyter."""

import contextlib
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import Mock

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
        self.assertEqual(controls["continuos"].options, ())
        self.assertTrue(controls["continuos"].disabled)
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
        output = self.click()
        self.assertEqual(self.state(self.panel.arbol), self.state(original_enriched))
        expected = io.StringIO()
        with contextlib.redirect_stdout(expected):
            print("Árbol enriquecido entrenado:")
            self.ns["print_tree"](self.panel.arbol, self.ns["columns"])
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
            max_depth=1, min_samples_split=3)
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
                            ("max_depth", -1), ("min_samples_split", 1)):
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
