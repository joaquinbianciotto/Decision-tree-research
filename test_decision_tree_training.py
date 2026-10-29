"""Comprueba la base preservada y el entrenamiento enriquecido aditivo."""

import ast
import contextlib
from copy import deepcopy
import io
import hashlib
import json
import inspect
from pathlib import Path
import unittest


ROOT = Path(__file__).parent


def execute(cells):
    namespace = {}
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        for cell in cells:
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    return namespace, stdout.getvalue(), stderr.getvalue()


class TrainingReviewNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
        self.cells = [cell for cell in self.notebook["cells"] if cell["cell_type"] == "code"]
        self.ns, self.stdout, self.stderr = execute(self.cells)

    def test_original_training_is_copied_without_integration(self):
        original = json.loads((ROOT / "decision-tree.ipynb").read_text())
        baseline = [cell for cell in original["cells"] if cell["cell_type"] == "code"][:4]
        training = next(cell for cell in self.cells if cell["id"] == "c45-algorithm")
        self.assertEqual(training["source"], baseline[-1]["source"])
        self.assertEqual(self.cells[-3], training)
        self.assertEqual(self.cells[-2]["id"], "integration-support")
        self.assertEqual(self.cells[-1]["id"], "enriched-training")
        for copied, source in zip((self.cells[i] for i in (0, 1, 2, 4)), baseline):
            self.assertEqual(copied["source"], source["source"])
        helper = next(c for c in self.cells if c["id"] == "proposed-extensions")
        self.assertEqual(hashlib.sha256("".join(helper["source"]).encode()).hexdigest(),
                         "da26a3319306592b52da8db0ce86d906323f3ecd47a701a3fa778fbaa98608ed")
        reference, stdout, stderr = execute(baseline)
        self.assertEqual((self.stdout, self.stderr), (stdout, stderr))
        features = [row[:-1] for row in reference["D"]]
        features.append(["snow", "cool", "high", "weak"])
        expected = [reference["predict_one"](reference["tree"], row) for row in features]
        actual = [self.ns["predict_one"](self.ns["tree"], row) for row in features]
        self.assertEqual(actual, expected)
        self.assertEqual(self.ns["D"], reference["D"])
        for cell in self.cells:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_enriched_core_is_separate_and_directly_recursive_like_baseline(self):
        by_id = {cell["id"]: cell for cell in self.cells}
        self.assertIn("integration-support", by_id)
        support = by_id["integration-support"]
        core = by_id["enriched-training"]
        self.assertLess(self.cells.index(support), self.cells.index(core))
        support_ast = ast.parse("".join(support["source"]))
        core_ast = ast.parse("".join(core["source"]))
        support_names = {node.name for node in support_ast.body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertTrue({"NodoEnriquecido", "_seleccionar_prueba_enriquecida",
                         "_recalibrar_enriquecido", "_podar_enriquecido",
                         "predecir_probabilidades_enriquecido", "predecir_enriquecido"}
                        <= support_names)
        functions = {node.name: node for node in core_ast.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(functions), {"_crecer_enriquecido", "entrenar_arbol_enriquecido"})
        grow = functions["_crecer_enriquecido"]
        self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.Lambda))
                             for node in ast.walk(grow) if node is not grow))
        first_stop = next(node for node in grow.body if isinstance(node, ast.If))
        self.assertIsInstance(first_stop.orelse[0], ast.If)
        calls = {node.func.id for node in ast.walk(grow)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue({"_seleccionar_prueba_enriquecida", "particionar_con_desconocidos",
                         "_crecer_enriquecido"} <= calls)
        self.assertFalse({"evaluar_umbrales_continuos", "evaluar_prueba", "_podar_enriquecido"} & calls)
        for loop in (node for node in ast.walk(grow) if isinstance(node, ast.For)):
            self.assertFalse(any(isinstance(child, ast.For)
                                 for statement in loop.body for child in ast.walk(statement)))

    def test_proposed_names_follow_book_notation_or_spanish(self):
        cell = next(cell for cell in self.cells if cell["id"] == "proposed-extensions")
        functions = {node.name for node in ast.parse("".join(cell["source"])).body
                     if isinstance(node, ast.FunctionDef)}
        self.assertEqual(functions, {
            "es_desconocido", "pesos_casos", "frecuencias_clase", "probabilidades_clase",
            "info", "separar_casos_conocidos", "evaluar_prueba",
            "particionar_con_desconocidos", "evaluar_umbrales_continuos",
            "combinar_probabilidades_clase", "errores_estimados", "U_CF",
            "estadisticas_hoja", "elegir_poda",
        })

    def test_markdown_is_spanish(self):
        titles = ["".join(cell["source"]).splitlines()[0]
                  for cell in self.notebook["cells"] if cell["cell_type"] == "markdown"]
        self.assertEqual(titles, [
            "# Entrenamiento del árbol de decisión — original y enriquecido",
            "## 1. Datos de entrada", "## 2. Estructura del árbol",
            "## 3. Evaluación de impureza",
            "## 4. Funciones aprobadas y fuentes",
            "## 5. Entrenamiento original — sin cambios",
            "## 6. Entrenamiento enriquecido — integración aprobada",
            "### 6.2 Crecimiento comparable con el algoritmo base",
        ])

    def test_every_helper_documents_pdf_and_both_page_numbers(self):
        cell = next(cell for cell in self.cells if cell["id"] == "proposed-extensions")
        for function in ast.parse("".join(cell["source"])).body:
            if isinstance(function, ast.FunctionDef):
                with self.subTest(function=function.name):
                    doc = ast.get_docstring(function) or ""
                    self.assertRegex(doc, r"Fuente conceptual: (NUL|CON|POD), libro pp?\.")
                    self.assertRegex(doc, r"PDF pp?\. [0-9]")
        markdown = "\n".join("".join(c["source"]) for c in self.notebook["cells"]
                             if c["cell_type"] == "markdown")
        for filename in (
            "(atributos continuos) Quinlan - C4_5_ Programs for Machine Learning.pdf",
            "(valores nulos) Quinlan - C4_5_ Programs for Machine Learning.pdf",
            "(poda)C4_5_ Programs for Machine Learning.pdf",
        ):
            self.assertIn(filename, markdown)
        self.assertIn("aproximación", self.ns["U_CF"].__doc__)

    def test_U_CF_is_a_rate_not_an_error_count(self):
        rate = self.ns["U_CF"]
        self.assertAlmostEqual(rate(0, 6), 0.206, places=3)
        self.assertEqual(rate(0, 1), 0.75)
        self.assertAlmostEqual(rate(1, 16), 0.157, places=3)
        self.assertAlmostEqual(16 * rate(1, 16), self.ns["errores_estimados"](16, 1))
        self.assertEqual(rate(0, 0), 0)
        with self.assertRaises(ValueError):
            rate(3, 2)

    def test_continuous_candidates_use_binary_splits_and_training_values(self):
        candidates = self.ns["evaluar_umbrales_continuos"]
        rows = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"]]
        before = deepcopy(rows)
        splits = candidates(rows, 0, reference_values=[1, 2, 3, 4, 5])
        self.assertEqual([split["threshold"] for split in splits], [3])
        self.assertEqual(splits[0]["branch_weights"], {True: 2.0, False: 2.0})
        self.assertEqual(splits[0]["gain"], 1.0)
        self.assertEqual(candidates(rows, 0)[0]["threshold"], 2)
        self.assertEqual(candidates([[1, "no"], [1, "yes"]], 0), [])
        self.assertEqual(candidates([[1, "no"], [2, "yes"]], 0), [])
        self.assertEqual(candidates([[None, "no"], ["?", "yes"]], 0), [])
        self.assertEqual(rows, before)

    def test_missing_values_adjust_gain_and_preserve_fractional_mass(self):
        es_desconocido = self.ns["es_desconocido"]
        self.assertTrue(all(es_desconocido(value) for value in (None, "?", float("nan"))))
        self.assertFalse(any(es_desconocido(value) for value in (0, False, "", "sunny")))
        rows = deepcopy(self.ns["D"])
        rows[2][0] = "?"
        before = deepcopy(rows)
        stats = self.ns["evaluar_prueba"](rows, 0)
        self.assertAlmostEqual(stats["gain"], 0.199, places=3)
        self.assertAlmostEqual(stats["split_info"], 1.809, places=3)
        self.assertAlmostEqual(stats["gain_ratio"], 0.110, places=3)
        groups, probabilities = self.ns["particionar_con_desconocidos"](rows, 0)
        self.assertEqual(probabilities, {"sunny": 5 / 13, "overcast": 3 / 13, "rain": 5 / 13})
        self.assertAlmostEqual(sum(sum(weights) for _, weights in groups.values()), 14)
        self.assertEqual(rows, before)
        weighted = [["a", "no"], ["b", "yes"], [None, "yes"]]
        weights = [2, 6, 4]
        weighted_before = deepcopy(weighted)
        groups, probabilities = self.ns["particionar_con_desconocidos"](weighted, 0, weights)
        self.assertEqual(probabilities, {"a": 0.25, "b": 0.75})
        self.assertEqual(groups["a"][1], [2.0, 1.0])
        self.assertEqual(groups["b"][1], [6.0, 3.0])
        self.assertEqual(weighted, weighted_before)
        self.assertEqual(weights, [2, 6, 4])
        self.assertEqual(self.ns["particionar_con_desconocidos"]([[None, "yes"]], 0), ({}, {}))
        self.assertIsNone(self.ns["evaluar_prueba"]([[None, "yes"]], 0))
        numeric = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"], [float("nan"), "yes"]]
        self.assertAlmostEqual(self.ns["evaluar_umbrales_continuos"](numeric, 0)[0]["gain"], 0.8)

    def test_missing_prediction_combines_class_distributions(self):
        distributions = {"low": {"yes": 1.0}, "high": {"yes": 5 / 44, "no": 39 / 44}}
        probabilities = {"low": 26 / 70, "high": 44 / 70}
        before = deepcopy((distributions, probabilities))
        result = self.ns["combinar_probabilidades_clase"](distributions, probabilities)
        self.assertAlmostEqual(result["yes"], 31 / 70)
        self.assertAlmostEqual(result["no"], 39 / 70)
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertEqual((distributions, probabilities), before)

    def test_pruning_estimates_and_choices_match_the_book_example(self):
        estimate = self.ns["errores_estimados"]
        choose = self.ns["elegir_poda"]
        subtree = sum(estimate(size, 0) for size in (6, 9, 1))
        leaf = estimate(16, 1)
        self.assertAlmostEqual(subtree, 3.273, places=3)
        # The book multiplies the rounded rate 0.157 by 16 to get 2.512.
        self.assertAlmostEqual(leaf / 16, 0.157, places=3)
        self.assertAlmostEqual(leaf, 2.512, delta=16 * 0.0005)
        self.assertEqual(choose(subtree, leaf), "leaf")
        self.assertEqual(choose(1, 2, 0.5), "raise")
        self.assertEqual(choose(1, 1, 1), "keep")
        self.assertEqual(estimate(0, 0), 0)
        self.assertEqual(estimate(2, 2), 2)
        self.assertLessEqual(estimate(0.5, 0), estimate(0.5, 0.25))
        self.assertLessEqual(estimate(0.5, 0.25), 0.5)
        rows = [["democrat"]] * 15 + [["republican"]]
        stats = self.ns["estadisticas_hoja"](rows)
        self.assertEqual(stats["prediction"], "democrat")
        self.assertEqual(stats["observed_errors"], 1)
        self.assertAlmostEqual(stats["estimated_errors"], leaf)

    def test_weight_contract_and_weighted_leaf_statistics(self):
        rows = [["no"], ["yes"]]
        self.assertEqual(self.ns["probabilidades_clase"](rows, [1, 3]), {"no": 0.25, "yes": 0.75})
        self.assertEqual(self.ns["info"](rows, [0, 3]), 0)
        self.assertEqual(self.ns["estadisticas_hoja"](rows, [1, 3])["prediction"], "yes")
        for weights in ([1], [-1, 2], [float("nan"), 1], [1, float("inf")]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.ns["pesos_casos"](rows, weights)
        self.assertEqual(rows, [["no"], ["yes"]])
        for total, errors, confidence in ((2, 3, 0.25), (2, -1, 0.25), (2, 0, 0), (2, 0, 1)):
            with self.subTest(total=total, errors=errors, confidence=confidence), self.assertRaises(ValueError):
                self.ns["errores_estimados"](total, errors, confidence)


class EnrichedTrainingTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.ns, self.stdout, self.stderr = execute(cells)

    def train(self, rows, attributes, *args, **kwargs):
        before = repr((rows, attributes, args, kwargs))
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return self.ns["entrenar_arbol_enriquecido"](rows, attributes, *args, **kwargs)
        finally:
            self.assertEqual(repr((rows, attributes, args, kwargs)), before)
            self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))

    def probabilities(self, tree, row):
        result = self.ns["predecir_probabilidades_enriquecido"](tree, row)
        self.assertAlmostEqual(sum(result.values()), 1)
        return result

    def depth(self, tree):
        return 0 if tree.attribute is None else 1 + max(self.depth(c) for c in tree.children.values())

    def weighted_three_level_rows(self):
        rows = [[a, b, c, "yes" if a and (not b or c) else "no"]
                for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        return rows, [20, 20, 20, 20, 7.5, 7.5, 2.5, 2.5]

    def test_max_depth_starts_at_zero_and_reaches_descendants_with_or_without_pruning(self):
        rows, weights = self.weighted_three_level_rows()
        for maximum, expected in ((None, 3), (0, 0), (1, 1), (2, 2), (3, 3)):
            for prune in (False, True):
                with self.subTest(maximum=maximum, prune=prune):
                    tree = self.train(rows, [0, 1, 2], pesos=weights,
                                      max_depth=maximum, podar=prune)
                    self.assertLessEqual(self.depth(tree), expected)
                    if not prune:
                        self.assertEqual(self.depth(tree), expected)
                    self.assertEqual(tree.prediction, "no")
                    self.assertEqual(tree.total_weight, 100)
                    if maximum == 1 and not prune:
                        self.assertEqual(tree.children[1].prediction, "yes")
                        self.assertEqual(self.probabilities(tree, [1, None, None]),
                                         {"yes": .875, "no": .125})

    def test_min_samples_split_counts_node_rows_not_weight_or_leaf_size(self):
        rows = [["left", "a", "no"], ["left", "b", "yes"],
                ["right", "a", "yes"], ["right", "b", "yes"]]
        for minimum, expected_depth in ((2, 2), (3, 1), (4, 1), (5, 0)):
            for weight in (.1, 100):
                for prune in (False, True):
                    with self.subTest(minimum=minimum, weight=weight, prune=prune):
                        tree = self.train(rows, [0, 1], pesos=[weight] * 4,
                                          min_samples_split=minimum, podar=prune)
                        self.assertLessEqual(self.depth(tree), expected_depth)
                        if not prune:
                            self.assertEqual(self.depth(tree), expected_depth)
                            if minimum in (3, 4):
                                self.assertEqual(tree.children["left"].prediction, "no")
                                self.assertEqual(self.probabilities(tree, ["left", "b"]),
                                                 {"no": .5, "yes": .5})

    def test_limits_apply_to_reused_continuous_attributes_and_combine(self):
        rows = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"], [7, "no"], [8, "no"]]
        for maximum, minimum, expected in ((None, 4, 2), (None, 5, 1),
                                           (None, 7, 0), (1, 4, 1), (0, 2, 0)):
            with self.subTest(maximum=maximum, minimum=minimum):
                tree = self.train(rows, [0], continuos=[0], max_depth=maximum,
                                  min_samples_split=minimum, podar=False)
                self.assertEqual(self.depth(tree), expected)
                if expected:
                    self.assertEqual((tree.attribute, tree.threshold), (0, 2))
                if expected == 1:
                    self.assertEqual(tree.children[False].prediction, "yes")
        # El mínimo de peso por lado continuo sigue siendo independiente.
        self.assertIsNone(self.train(rows, [0], continuos=[0],
                                     min_samples_split=2, min_peso_rama=3.01,
                                     podar=False).attribute)

    def test_min_gain_uses_original_weight_and_allows_equality(self):
        rows = [["large", "a", "no"], ["large", "b", "no"],
                ["small", "a", "yes"], ["small", "b", "no"]]
        for scale in (.5, 1, 10):
            for minimum, split in ((.15, False), (.1, True)):
                with self.subTest(scale=scale, minimum=minimum):
                    tree = self.train(rows, [0, 1], pesos=[w * scale for w in (45, 45, 5, 5)],
                                      min_gain=minimum, podar=False)
                    self.assertEqual(tree.attribute, 0)
                    child = tree.children["small"]
                    self.assertEqual(child.attribute, 1 if split else None)
                    self.assertEqual(child.prediction, "yes")
                    self.assertAlmostEqual(child.total_weight, 10 * scale)
                    # En un empate, la predicción pública conserva el orden de clases de la raíz.
                    self.assertEqual(self.ns["predecir_enriquecido"](tree, ["small", "a"]),
                                     "yes" if split else "no")

    def test_original_weight_survives_three_levels_and_reinitializes_per_training(self):
        rows, weights = self.weighted_three_level_rows()
        for minimum, split in ((.055, False), (.05, True)):
            with self.subTest(minimum=minimum):
                tree = self.train(rows, [0, 1, 2], pesos=weights,
                                  min_gain=minimum, podar=False)
                self.assertEqual(tree.attribute, 0)
                self.assertEqual(tree.children[1].attribute, 1)
                grandchild = tree.children[1].children[1]
                self.assertEqual(grandchild.total_weight, 5)
                self.assertEqual(grandchild.attribute, 2 if split else None)
        independent = self.train(rows[-2:], [2], pesos=weights[-2:], min_gain=.055, podar=False)
        self.assertEqual((independent.total_weight, independent.attribute), (5, 2))

    def test_global_gain_preserves_missing_adjustment_and_fractional_node_weight(self):
        rows = [["large", "a", "no"], ["large", "b", "no"],
                ["small", "a", "yes"], ["small", "b", "no"],
                ["small", None, "yes"], [None, "?", "yes"]]
        weights = [45, 45, 4, 4, 2, 2]
        # El hijo pesa 10.2, pero sólo 8 son conocidos: ganancia global = 8/102.
        for minimum, split in ((.08, False), (.07, True)):
            with self.subTest(minimum=minimum):
                tree = self.train(rows, [0, 1], pesos=weights, min_gain=minimum, podar=False)
                self.assertEqual((tree.attribute, tree.total_weight), (0, 102))
                child = tree.children["small"]
                self.assertAlmostEqual(child.total_weight, 10.2)
                self.assertEqual(child.attribute, 1 if split else None)
                self.assertAlmostEqual(child.class_counts["yes"], 6.2)
                self.assertEqual(child.class_counts["no"], 4)
                if split:
                    self.assertEqual(child.branch_probabilities, {"a": .5, "b": .5})
                    for leaf in child.children.values():
                        self.assertAlmostEqual(leaf.total_weight, 5.1)

    def test_existing_positional_arguments_and_appended_defaults_remain_compatible(self):
        signature = str(inspect.signature(self.ns["entrenar_arbol_enriquecido"]))
        self.assertEqual(signature, "(filas, atributos, continuos=(), pesos=None, min_gain=1e-12, "
                         "min_peso_rama=2.0, confianza=0.25, podar=True, "
                         "max_depth=None, min_samples_split=2)")
        rows = [[1, "no"], [1, "no"], [2, "yes"], [2, "yes"]]
        positional = self.train(rows, [0], [0], [1] * 4, 1, 2.0, .25, False)
        keyword = self.train(rows, [0], continuos=[0], pesos=[1] * 4,
                             min_gain=1, min_peso_rama=2.0, confianza=.25, podar=False)
        self.assertEqual((positional.threshold, keyword.threshold), (1, 1))
        self.assertEqual(self.probabilities(positional, [None]), self.probabilities(keyword, [None]))

    def test_numeric_reuse_whole_training_reference_and_no_mutation(self):
        rows = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"],
                [7, "no"], [8, "no"], [3, "ignored"]]
        attributes, weights = [0], [1, 1, 1, 1, 1, 1, 0]
        before = deepcopy((rows, attributes, weights))
        tree = self.train(rows, attributes, continuos=[0], pesos=weights, podar=False)
        self.assertNotIsInstance(tree, self.ns["Node"])
        self.assertEqual((tree.attribute, tree.threshold), (0, 3))
        self.assertEqual((tree.children[False].attribute, tree.children[False].threshold), (0, 5))
        self.assertEqual([self.ns["predecir_enriquecido"](tree, [i]) for i in (1, 4, 8)],
                         ["no", "yes", "no"])
        self.assertEqual((rows, attributes, weights), before)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.ns["print_tree"](tree, ["Medición", "Clase"])
        self.assertIn("<= 3", output.getvalue())
        categorical = self.train(rows[:6], attributes, podar=False)
        self.assertIsNone(categorical.threshold)
        self.assertEqual(len(categorical.children), 6)

    def test_repeated_numbers_minimum_known_weight_and_gain_boundary(self):
        rows = [[1, "no"], [1, "no"], [2, "yes"], [2, "yes"]]
        tree = self.train(rows, [0], continuos=[0], podar=False, min_gain=1)
        self.assertEqual(tree.threshold, 1)
        self.assertIsNone(self.train(rows, [0], continuos=[0], min_gain=1.01).attribute)
        self.assertIsNone(self.train(rows, [0], continuos=[0], min_peso_rama=2.01).attribute)
        self.assertIsNone(self.train(rows, [0], continuos=[0], pesos=[1, 0, 1, 0]).attribute)
        self.assertIsNone(self.train([[1, "no"], [1, "yes"]], [0],
                                     continuos=[0], min_gain=0).attribute)

    def test_fractional_fit_missing_and_unseen_probability_prediction(self):
        rows, weights, attributes = [["a", "no"], ["b", "yes"], [None, "yes"]], [2, 6, 4], [0]
        before = deepcopy((rows, weights, attributes))
        tree = self.train(rows, attributes, pesos=weights, podar=False)
        self.assertEqual(tree.class_counts, {"no": 2, "yes": 10})
        self.assertEqual(tree.total_weight, 12)
        self.assertEqual(tree.branch_probabilities, {"a": .25, "b": .75})
        self.assertEqual(tree.children["a"].class_counts, {"no": 2, "yes": 1})
        self.assertEqual(tree.children["a"].total_weight, 3)
        self.assertEqual(tree.children["b"].total_weight, 9)
        for missing in (None, "?", float("nan")):
            distribution = self.probabilities(tree, [missing])
            self.assertAlmostEqual(distribution["no"], 1 / 6)
            self.assertAlmostEqual(distribution["yes"], 5 / 6)
        self.assertEqual(self.probabilities(tree, ["unseen"]), {"no": 1 / 6, "yes": 5 / 6})
        self.assertEqual(self.probabilities(tree, ["a"]), {"no": 2 / 3, "yes": 1 / 3})
        self.assertEqual((rows, weights, attributes), before)

    def test_weighted_continuous_missing_fit_and_boundary_predictions(self):
        rows = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"],
                [None, "yes"], ["?", "no"], [float("nan"), "yes"]]
        attributes, continuous = [0], [0]
        weights = [1.5, .5, 2, 4, .5, .25, .25]
        before = deepcopy((rows, attributes, continuous, weights))
        for prune in (False, True):
            with self.subTest(prune=prune):
                tree = self.train(rows, attributes, continuos=continuous, pesos=weights, podar=prune)
                self.assertEqual((tree.attribute, tree.threshold, tree.total_weight), (0, 2, 9))
                self.assertEqual(tree.branch_probabilities, {True: 1 / 4, False: 3 / 4})
                self.assertEqual(tree.children[True].class_counts, {"no": 33 / 16, "yes": 3 / 16})
                self.assertEqual(tree.children[False].class_counts, {"yes": 105 / 16, "no": 3 / 16})
                # El umbral pertenece al lado izquierdo; 3 es nuevo y cae a la derecha.
                for query, expected, label in (
                    ([2], {"no": 11 / 12, "yes": 1 / 12}, "no"),
                    ([3], {"no": 1 / 36, "yes": 35 / 36}, "yes"),
                    ([float("nan")], {"no": 1 / 4, "yes": 3 / 4}, "yes"),
                ):
                    query_before = deepcopy(query)
                    actual = self.probabilities(tree, query)
                    self.assertEqual(set(actual), set(expected))
                    for class_name, probability in expected.items():
                        self.assertAlmostEqual(actual[class_name], probability)
                    self.assertEqual(self.ns["predecir_enriquecido"](tree, query), label)
                    self.assertEqual(query, query_before)
                self.assertEqual((rows, attributes, continuous, weights), before)

    def test_multilevel_missing_prediction_uses_conditional_branch_products(self):
        rows = [["x", "u", "no"], ["x", "v", "yes"],
                ["y", "u", "yes"], ["y", "v", "no"]]
        attributes, weights = [0, 1], [6, 2, 9, 1]
        before = deepcopy((rows, attributes, weights))
        for prune in (False, True):
            with self.subTest(prune=prune):
                tree = self.train(rows, attributes, pesos=weights, podar=prune)
                self.assertEqual(tree.attribute, 0)
                self.assertEqual(tree.branch_probabilities, {"x": 4 / 9, "y": 5 / 9})
                self.assertEqual([child.attribute for child in tree.children.values()], [1, 1])
                self.assertEqual(tree.children["x"].branch_probabilities, {"u": 3 / 4, "v": 1 / 4})
                self.assertEqual(tree.children["y"].branch_probabilities, {"u": 9 / 10, "v": 1 / 10})
                # P(no | ?, ?) = (4/9)(3/4) + (5/9)(1/10), no una media uniforme.
                for query, no_probability, label in (
                    ([None, "?"], 7 / 18, "yes"),
                    ([None, "u"], 4 / 9, "yes"),
                    ([None, "v"], 5 / 9, "no"),
                    (["x", None], 3 / 4, "no"),
                    (["y", None], 1 / 10, "yes"),
                ):
                    query_before = deepcopy(query)
                    actual = self.probabilities(tree, query)
                    self.assertEqual(set(actual), {"no", "yes"})
                    self.assertAlmostEqual(actual["no"], no_probability)
                    self.assertAlmostEqual(actual["yes"], 1 - no_probability)
                    self.assertEqual(self.ns["predecir_enriquecido"](tree, query), label)
                    self.assertEqual(query, query_before)
                self.assertEqual((rows, attributes, weights), before)

    def test_multilevel_unknown_training_case_preserves_fractional_class_mass(self):
        rows = [["x", "u", "no"], ["x", "v", "yes"],
                ["y", "u", "yes"], ["y", "v", "no"], [None, "?", "yes"]]
        attributes, weights = [0, 1], [6, 2, 9, 1, 2]
        before = deepcopy((rows, attributes, weights))
        tree = self.train(rows, attributes, pesos=weights, podar=False)
        self.assertEqual((tree.attribute, tree.total_weight, tree.class_counts),
                         (0, 20, {"no": 7, "yes": 13}))
        self.assertEqual([child.attribute for child in tree.children.values()], [1, 1])
        # El caso de peso 2 llega a x/u con 2*(4/9)*(3/4) = 2/3, no con 2.
        expected_leaves = {
            ("x", "u"): {"no": 6, "yes": 2 / 3},
            ("x", "v"): {"yes": 20 / 9},
            ("y", "u"): {"yes": 10},
            ("y", "v"): {"no": 1, "yes": 1 / 9},
        }
        totals = {"no": 0, "yes": 0}
        for (first, second), counts in expected_leaves.items():
            leaf = tree.children[first].children[second]
            self.assertIsNone(leaf.attribute)
            self.assertEqual(set(leaf.class_counts), set(counts))
            for label, count in counts.items():
                self.assertAlmostEqual(leaf.class_counts[label], count)
                totals[label] += leaf.class_counts[label]
            self.assertAlmostEqual(leaf.total_weight, sum(counts.values()))
        self.assertAlmostEqual(totals["no"], 7)
        self.assertAlmostEqual(totals["yes"], 13)
        for query, no_probability, label in (([None, None], 7 / 20, "yes"),
                                              ([None, "u"], 2 / 5, "yes"),
                                              (["x", "u"], 9 / 10, "no")):
            actual = self.probabilities(tree, query)
            self.assertAlmostEqual(actual["no"], no_probability)
            self.assertAlmostEqual(actual["yes"], 1 - no_probability)
            self.assertEqual(self.ns["predecir_enriquecido"](tree, query), label)
        self.assertEqual((rows, attributes, weights), before)

    def test_public_training_rejects_invalid_weights_without_side_effects(self):
        rows, attributes = [[1, "no"], [2, "yes"]], [0]
        for weights in ([1], [1, 1, 1], [-.5, 1], [float("nan"), 1],
                        [1, float("inf")], [float("-inf"), 1]):
            for prune in (False, True):
                with self.subTest(weights=weights, prune=prune):
                    before = deepcopy((rows, attributes, weights))
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        with self.assertRaises(ValueError) as raised:
                            self.ns["entrenar_arbol_enriquecido"](
                                filas=rows, atributos=attributes, pesos=weights, podar=prune)
                    self.assertIs(type(raised.exception), ValueError)
                    self.assertEqual(str(raised.exception), "Provide one finite nonnegative weight per row.")
                    self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))
                    self.assertEqual((rows, attributes, weights), before)

    def test_authorized_missing_prediction_example_through_public_training(self):
        rows = [["low", "yes"], ["high", "yes"], ["high", "no"]]
        tree = self.train(rows, [0], pesos=[26, 5, 39], podar=False)
        result = self.probabilities(tree, [None])
        self.assertAlmostEqual(result["yes"], 31 / 70)
        self.assertAlmostEqual(result["no"], 39 / 70)
        self.assertEqual(self.ns["predecir_enriquecido"](tree, [None]), "no")

    def test_no_known_values_purity_majority_and_selection_ties(self):
        rows = [[None, "first"], ["?", "second"]]
        tree = self.train(rows, [0], pesos=[1, 3], min_gain=0)
        self.assertIsNone(tree.attribute)
        self.assertEqual(tree.prediction, "second")
        self.assertEqual(self.probabilities(tree, [None]), {"first": .25, "second": .75})
        self.assertEqual(self.train(rows, [0]).prediction, "first")
        self.assertIsNone(self.train([[1, "a"], [2, "b"]], [0], pesos=[1, 0]).attribute)
        tied = [["x", "x", "first"], ["y", "y", "second"]]
        tree = self.train(tied, [1, 0], podar=False)
        self.assertEqual(tree.attribute, 1)
        self.assertEqual(self.ns["predecir_enriquecido"](tree, [None, None]), "first")
        self.assertTrue(all(c.attribute is None for c in tree.children.values()))
        xor = [[0, 0, "a"], [0, 1, "b"], [1, 0, "b"], [1, 1, "a"]]
        self.assertIsNone(self.train(xor, [0, 1], min_gain=0).attribute)
        empty = self.train(rows, [0], pesos=[0, 0])
        self.assertIsNone(empty.prediction)
        self.assertEqual(self.ns["predecir_probabilidades_enriquecido"](empty, [None]), {})

    def test_quinlan_leaf_replacement_is_real_and_original_is_unchanged(self):
        rows = [["a", "democrat"]] * 6 + [["b", "democrat"]] * 9 + [["c", "republican"]]
        original = self.ns["tree"]
        baseline = [self.ns["predict_one"](original, row[:-1]) for row in self.ns["D"]]
        unpruned = self.train(rows, [0], podar=False)
        pruned = self.train(rows, [0])
        self.assertEqual(len(unpruned.children), 3)
        self.assertIsNone(pruned.attribute)
        self.assertEqual(pruned.prediction, "democrat")
        self.assertEqual(pruned.total_weight, 16)
        self.assertEqual(pruned.class_counts, {"democrat": 15, "republican": 1})
        self.assertEqual(self.train(rows, [0], confianza=.99).attribute, 0)
        self.assertIs(self.ns["tree"], original)
        self.assertEqual([self.ns["predict_one"](original, row[:-1]) for row in self.ns["D"]], baseline)

    def raising_rows(self):
        rows = [["main", 0, "no"]] * 4 + [["main", 1, "yes"]] * 4
        for i, (value, label) in enumerate([(0, "no"), (0, "yes"), (1, "no"), (1, "yes")] * 2):
            rows.append([str(i), value, label])
        return rows

    def test_branch_raising_reroutes_and_recalibrates_full_parent_population(self):
        rows = self.raising_rows()
        before = deepcopy(rows)
        unpruned = self.train(rows, [0, 1], podar=False)
        pruned = self.train(rows, [0, 1])
        self.assertEqual(unpruned.attribute, 0)
        self.assertEqual(unpruned.children["main"].attribute, 1)
        self.assertEqual(pruned.attribute, 1)
        self.assertEqual(pruned.total_weight, 16)
        self.assertEqual(pruned.class_counts, {"no": 8, "yes": 8})
        self.assertEqual(pruned.branch_probabilities, {0: .5, 1: .5})
        self.assertEqual(pruned.children[0].total_weight, 8)
        self.assertEqual(pruned.children[0].class_counts, {"no": 6, "yes": 2})
        self.assertEqual(pruned.children[1].class_counts, {"yes": 6, "no": 2})
        self.assertEqual(self.probabilities(pruned, ["unseen", 0]), {"no": .75, "yes": .25})
        estimate = self.ns["errores_estimados"]
        self.assertLess(2 * estimate(8, 2), 2 * estimate(4, 0) + 8 * estimate(1, 0))
        self.assertLess(2 * estimate(8, 2), estimate(16, 8))
        self.assertEqual(rows, before)

    def test_raising_with_unknowns_preserves_mass_and_normalization(self):
        rows = self.raising_rows() + [["?", None, "yes"]]
        weights = [1] * 16 + [.5]
        before = deepcopy((rows, weights))
        tree = self.train(rows, [0, 1], pesos=weights)
        self.assertEqual(tree.attribute, 1)
        self.assertAlmostEqual(tree.total_weight, 16.5)
        self.assertAlmostEqual(sum(c.total_weight for c in tree.children.values()), 16.5)
        self.assertAlmostEqual(tree.children[0].total_weight, 8.25)
        self.assertAlmostEqual(tree.children[0].class_counts["yes"], 2.25)
        self.probabilities(tree, [None, None])
        self.assertEqual((rows, weights), before)

    def test_raising_is_rejected_when_its_old_subset_hides_parent_errors(self):
        rows = ([["main", 0, "no"]] * 20 + [["main", 1, "yes"]] * 10
                + [["other", 0, "yes"]] * 10)
        tree = self.train(rows, [0, 1])
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(tree.children["main"].attribute, 1)
        self.assertEqual(tree.total_weight, 40)
        self.assertEqual(self.probabilities(tree, ["other", 0]), {"yes": 1})
        estimate = self.ns["errores_estimados"]
        old_subset = estimate(20, 0) + estimate(10, 0)
        subtree = old_subset + estimate(10, 0)
        full_population = estimate(30, 10) + estimate(10, 0)
        self.assertLess(old_subset, subtree)
        self.assertGreater(full_population, subtree)

    def test_raised_categorical_fallback_is_counted_on_full_population(self):
        rows = [["main", 0, "no"]] * 4 + [["main", 1, "yes"]] * 4
        rows += [[str(i), 2, "yes" if i % 2 else "no"] for i in range(8)]
        tree = self.train(rows, [0, 1])
        self.assertEqual(tree.attribute, 1)
        self.assertEqual(tree.total_weight, 16)
        self.assertEqual(sum(c.total_weight for c in tree.children.values()), 8)
        self.assertEqual(tree.branch_probabilities, {0: .5, 1: .5})
        self.assertEqual(self.probabilities(tree, ["outside", 2]), {"no": .5, "yes": .5})
        self.assertEqual(self.ns["predecir_enriquecido"](tree, ["outside", 2]), "no")

    def test_enriched_execution_preserves_in_memory_baseline_and_helpers(self):
        notebook = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        before_source = deepcopy(cells[:-1])
        namespace, stdout, stderr = execute(cells[:-1])
        old_bindings = dict(namespace)
        def tree_state(node):
            return (node.prediction, node.attribute, node.threshold,
                    {key: tree_state(child) for key, child in node.children.items()})
        old_state = tree_state(namespace["tree"])
        old_data = deepcopy((namespace["D"], namespace["A"], namespace["training_matrix"]))
        extra_stdout, extra_stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(extra_stdout), contextlib.redirect_stderr(extra_stderr):
            exec(compile("".join(cells[-1]["source"]), cells[-1]["id"], "exec"), namespace)
        self.assertEqual((extra_stdout.getvalue(), extra_stderr.getvalue()), ("", ""))
        self.assertEqual((stdout, stderr), (self.stdout, self.stderr))
        for name, value in old_bindings.items():
            self.assertIs(namespace[name], value, name)
        self.assertEqual(tree_state(namespace["tree"]), old_state)
        self.assertEqual((namespace["D"], namespace["A"], namespace["training_matrix"]), old_data)
        self.assertEqual(cells[:-1], before_source)
        tree = namespace["arbol_enriquecido"]
        distribution = namespace["predecir_probabilidades_enriquecido"](tree, [None] * 4)
        distribution.clear()
        self.assertTrue(tree.class_probabilities)

    def test_public_api_matches_pre_refactor_behavioral_snapshot(self):
        snapshot = public_api_snapshot()
        self.assertEqual(len(snapshot["cases"]), 46)
        # Los casos anteriores no cambian; la firma sólo agrega dos opciones al final.
        suffix = ", max_depth=None, min_samples_split=2)"
        signature = snapshot["signatures"]["entrenar_arbol_enriquecido"]
        self.assertTrue(signature.endswith(suffix))
        snapshot["signatures"]["entrenar_arbol_enriquecido"] = signature[:-len(suffix)] + ")"
        digest = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(digest, "2bacb63d62d834d2ac49408c3e39b975e89d1e042ea141b9b74d478c9b72d481")

    def test_run_all_demo_is_silent_and_keeps_baseline_data(self):
        tree = self.ns["arbol_enriquecido"]
        self.assertEqual(tree.total_weight, 14)
        self.assertEqual((tree.attribute, tree.threshold, tree.prediction), (0, None, "yes"))
        self.assertEqual(self.stderr, "")
        self.assertEqual(self.ns["A"], [0, 1, 2, 3])
        self.assertEqual(self.probabilities(tree, [None] * 4), {"no": 5 / 14, "yes": 9 / 14})


def public_api_snapshot():
    """Captura observable completa para comparar antes/después sin guardar implementaciones."""
    notebook = json.loads((ROOT / "decision-tree-training.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    namespace, stdout, stderr = execute(cells)
    def state(tree):
        return (tree.prediction, tree.attribute, tree.threshold, list(tree.class_counts.items()),
                tree.total_weight, list(tree.class_probabilities.items()),
                list(tree.branch_probabilities.items()),
                [(branch, state(child)) for branch, child in tree.children.items()])
    numeric = [[1, "no"], [2, "no"], [4, "yes"], [5, "yes"],
               [7, "no"], [8, "no"], [3, "ignored"]]
    missing = [["a", "no"], ["b", "yes"], [None, "yes"], ["?", "no"], [float("nan"), "yes"]]
    leaf = [["a", "democrat"]] * 6 + [["b", "democrat"]] * 9 + [["c", "republican"]]
    raising = EnrichedTrainingTests().raising_rows()
    rejected = ([["main", 0, "no"]] * 20 + [["main", 1, "yes"]] * 10
                + [["other", 0, "yes"]] * 10)
    fallback = [["main", 0, "no"]] * 4 + [["main", 1, "yes"]] * 4
    fallback += [[str(i), 2, "yes" if i % 2 else "no"] for i in range(8)]
    cases = [
        (numeric, [0], {"continuos": [0], "pesos": [1] * 6 + [0]}, [[1], [3], [4], [8], [None]]),
        (numeric[:6], [0], {}, [[1], [4], [9], [None]]),
        (missing, [0], {"pesos": [2, 6, 4, .25, .5]}, [["a"], ["b"], ["new"], [None], [float("nan")]]),
        ([[None, "first"], ["?", "second"]], [0], {"pesos": [1, 3]}, [[None], ["new"]]),
        ([[None, "first"], ["?", "second"]], [0], {"pesos": [0, 0]}, [[None]]),
        ([["x", "x", "first"], ["y", "y", "second"]], [1, 0], {}, [[None, None], ["x", "new"]]),
        ([[0, 0, "a"], [0, 1, "b"], [1, 0, "b"], [1, 1, "a"]], [0, 1], {"min_gain": 0}, [[0, 1]]),
        (leaf, [0], {}, [["a"], ["new"], [None]]),
        (raising, [0, 1], {}, [["main", 0], ["new", 1], [None, None], ["new", 2]]),
        (rejected, [0, 1], {}, [["other", 0], [None, None], ["new", 2]]),
        (raising + [["?", None, "yes"]], [0, 1], {"pesos": [1] * 16 + [.5]}, [[None, None], ["new", 0]]),
        (fallback, [0, 1], {}, [["outside", 2], [None, None]]),
        ([[1, "a"], [2, "b"]], [], {}, [[1], [None]]),
        ([[1, "a"], [2, "b"]], [0], {"pesos": [1, 0]}, [[2]]),
        ([[1, "a"], [1, "b"]], [0], {"continuos": [0], "min_gain": 0}, [[1], [None]]),
    ]
    boundary = [[1, "no"], [1, "no"], [2, "yes"], [2, "yes"]]
    for gain in (0, 1, 1.01, float("nan")):
        cases.append((boundary, [0], {"continuos": [0], "min_gain": gain}, [[1], [2], [None]]))
    for minimum in (2, 2.01):
        cases.append((boundary, [0], {"continuos": [0], "min_peso_rama": minimum}, [[1], [2], [None]]))
    for confidence in (.25, .99):
        cases.append((leaf, [0], {"confianza": confidence}, [["a"], ["c"], [None]]))
    observations = []
    for rows, attributes, options, queries in cases:
        for prune in (False, True):
            rows, attributes, options = deepcopy((rows, attributes, options))
            before = repr((rows, attributes, options, queries))
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                tree = namespace["entrenar_arbol_enriquecido"](
                    filas=rows, atributos=attributes, **options, podar=prune)
                probabilities = [list(namespace["predecir_probabilidades_enriquecido"](
                    arbol=tree, fila=row).items()) for row in queries]
                classes = [namespace["predecir_enriquecido"](arbol=tree, fila=row) for row in queries]
                namespace["print_tree"](tree, ["Atributo 0", "Atributo 1", "Clase"])
            assert repr((rows, attributes, options, queries)) == before
            observations.append((0, output.getvalue(), errors.getvalue(), state(tree), probabilities, classes))
    signatures = {name: str(inspect.signature(namespace[name])) for name in (
        "entrenar_arbol_enriquecido", "predecir_probabilidades_enriquecido", "predecir_enriquecido")}
    return {"execution": (0, stdout, stderr), "signatures": signatures,
            "demo": state(namespace["arbol_enriquecido"]), "cases": observations}


if __name__ == "__main__":
    unittest.main()
