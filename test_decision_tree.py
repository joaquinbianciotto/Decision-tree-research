"""Test the notebook's literal recursive decision-tree algorithm.

Run with: python -m unittest -v test_decision_tree.py
"""

import ast
import contextlib
import io
import json
from pathlib import Path
import unittest


NOTEBOOK = Path(__file__).with_name("decision-tree.ipynb")


class DecisionTreeNotebookTests(unittest.TestCase):
    def setUp(self):
        source = NOTEBOOK.read_text(encoding="utf-8")
        self.assertTrue(source.strip(), "The notebook must contain valid notebook JSON.")
        self.notebook = json.loads(source)
        self.cells = [
            cell for cell in self.notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.namespace = {}
        self.outputs = []
        for index, cell in enumerate(self.cells, start=1):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(
                    compile("".join(cell["source"]), f"notebook-cell-{index}", "exec"),
                    self.namespace,
                )
            self.outputs.append(output.getvalue())

    def train(self, rows, attributes, **options):
        before_rows = [row.copy() for row in rows]
        before_attributes = attributes.copy()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            tree = self.namespace["decision_tree"](rows, attributes, **options)
        self.assertEqual(rows, before_rows)
        self.assertEqual(attributes, before_attributes)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        return tree

    def test_five_cells_and_library_constraints(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(len(self.cells), 5)
        for index, cell in enumerate(self.cells):
            imports = [
                node
                for node in ast.walk(ast.parse("".join(cell["source"])))
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            if index in (1, 3):
                self.assertEqual(imports, [])
            for node in imports:
                modules = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                self.assertTrue(set(modules) <= {"collections", "math"})

    def test_labeled_training_matrix_has_class_last(self):
        matrix = self.namespace["training_matrix"]
        self.assertEqual(matrix[0], self.namespace["columns"])
        self.assertEqual(matrix[0][-1], "Play")
        self.assertEqual(matrix[1:], self.namespace["D"])
        self.assertTrue(all(len(row) == len(matrix[0]) for row in matrix[1:]))
        self.assertEqual({row[-1] for row in matrix[1:]}, {"yes", "no"})

    def test_node_is_a_simple_leaf_or_branch(self):
        node = self.namespace["Node"]("yes")
        self.assertEqual(node.prediction, "yes")
        self.assertIsNone(node.attribute)
        self.assertIsNone(node.threshold)
        self.assertEqual(node.children, {})

    def test_entropy_and_weighted_conditional_entropy(self):
        entropy = self.namespace["impurityEval_1"]
        conditional = self.namespace["impurityEval_2"]
        self.assertEqual(entropy([]), 0.0)
        self.assertEqual(entropy([["yes"], ["yes"]]), 0.0)
        self.assertEqual(entropy([["yes"], ["no"]]), 1.0)
        self.assertAlmostEqual(entropy(self.namespace["D"]), 0.9402859586706311)
        self.assertAlmostEqual(conditional(0, self.namespace["D"]), 0.6935361388961918)
        self.assertEqual(conditional(0, []), 0.0)
        self.assertEqual(conditional(0, [[1, "no"], [2, "yes"]], 1.5), 0.0)

    def test_training_uses_direct_recursion_without_nested_helpers(self):
        module = ast.parse("".join(self.cells[3]["source"]))
        training = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == "decision_tree")
        self.assertFalse(any(isinstance(node, ast.FunctionDef)
                             for statement in training.body for node in ast.walk(statement)))
        self.assertFalse(any(isinstance(node, (ast.Raise, ast.Assert))
                             for node in ast.walk(training)))
        self.assertFalse(any(isinstance(node, ast.Name) and node.id == "grow"
                             for node in ast.walk(module)))
        calls = [node for node in ast.walk(training)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "decision_tree"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([argument.id for argument in calls[0].args],
                         ["group", "remaining", "min_gain"])
        helpers = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"predict_one", "print_tree"} <= helpers)

    def test_information_gain_not_gain_ratio_selects_attribute(self):
        rows = [
            [f"id-{i}", "low" if i < 4 else "high", "no" if i < 4 else "yes"]
            for i in range(8)
        ]
        tree = self.train(rows, [0, 1])
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(set(tree.children), {row[0] for row in rows})
        self.assertTrue(all(child.attribute is None for child in tree.children.values()))
        self.assertEqual(
            [self.namespace["predict_one"](tree, row[:-1]) for row in rows],
            [row[-1] for row in rows],
        )
        self.assertEqual(self.train(rows, [1, 0]).attribute, 1)

    def test_largest_impurity_reduction_wins(self):
        rows = (
            [["left", "common", "yes"] for _ in range(30)]
            + [["left", "common", "no"] for _ in range(20)]
            + [["right", "common", "yes"] for _ in range(20)]
            + [["right", "common", "no"] for _ in range(30)]
        )
        rows[0][1] = "rare"
        tree = self.train(rows, [0, 1])
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(set(tree.children), {"left", "right"})

    def test_numbers_form_value_branches_without_thresholds(self):
        rows = [[1, "no"], [2, "no"], [3, "yes"], [4, "yes"], [5, "no"], [6, "no"]]
        tree = self.train(rows, [0])
        self.assertEqual(tree.attribute, 0)
        self.assertIsNone(tree.threshold)
        self.assertEqual(set(tree.children), {1, 2, 3, 4, 5, 6})
        self.assertTrue(all(child.attribute is None for child in tree.children.values()))
        self.assertEqual(
            [self.namespace["predict_one"](tree, row[:-1]) for row in rows],
            [row[-1] for row in rows],
        )

    def test_stopping_rules_and_unseen_category_fallback(self):
        predict = self.namespace["predict_one"]
        examples = [
            ([["a", "yes"], ["b", "yes"]], [0], "yes"),
            ([["a", "no"], ["b", "yes"], ["c", "yes"]], [], "yes"),
            ([["a", "no"], ["a", "yes"], ["a", "yes"]], [0], "yes"),
        ]
        for rows, attributes, majority in examples:
            with self.subTest(rows=rows):
                tree = self.train(rows, attributes)
                self.assertIsNone(tree.attribute)
                self.assertEqual(tree.prediction, majority)
                self.assertEqual(tree.children, {})
        tree = self.train([[1, "no"], [2, "yes"]], [0], min_gain=2)
        self.assertIsNone(tree.attribute)
        self.assertEqual(tree.prediction, "no")
        self.assertEqual(tree.children, {})
        self.assertEqual(predict(self.namespace["tree"], ["snow", "cool", "high", "weak"]), "yes")
        self.assertEqual(
            [predict(self.namespace["tree"], row[:-1]) for row in self.namespace["D"]],
            [row[-1] for row in self.namespace["D"]],
        )

    def test_gain_equal_to_threshold_still_splits(self):
        rows = [["a", "no"], ["b", "yes"]]
        tree = self.train(rows, [0], min_gain=1.0)
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(set(tree.children), {"a", "b"})
        self.assertEqual(tree.children["a"].prediction, "no")
        self.assertEqual(tree.children["b"].prediction, "yes")
        self.assertTrue(all(child.attribute is None for child in tree.children.values()))

    def test_zero_gain_split_recurses_without_reusing_attributes(self):
        rows = [[1, "a", "no"], [1, "b", "yes"],
                [2, "a", "yes"], [2, "b", "no"]]
        tree = self.train(rows, [0, 1], min_gain=0)
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(set(tree.children), {1, 2})
        for child in tree.children.values():
            self.assertEqual(child.attribute, 1)
            self.assertEqual(set(child.children), {"a", "b"})
            self.assertTrue(all(leaf.attribute is None for leaf in child.children.values()))
        self.assertEqual(
            [self.namespace["predict_one"](tree, row[:-1]) for row in rows],
            [row[-1] for row in rows],
        )

    def test_constant_attribute_at_zero_threshold_is_removed(self):
        tree = self.train([["a", "no"], ["a", "yes"], ["a", "yes"]],
                          [0], min_gain=0)
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(set(tree.children), {"a"})
        child = tree.children["a"]
        self.assertIsNone(child.attribute)
        self.assertEqual(child.prediction, "yes")
        self.assertEqual(child.children, {})

    def test_final_cell_reports_only_held_out_accuracy(self):
        matrix = self.namespace["test_matrix"]
        self.assertEqual(matrix[0], self.namespace["columns"])
        training = {tuple(row) for row in self.namespace["D"]}
        self.assertTrue(training.isdisjoint(tuple(row) for row in matrix[1:]))
        predictions = [
            self.namespace["predict_one"](self.namespace["tree"], row[:-1])
            for row in matrix[1:]
        ]
        self.assertEqual(predictions, [row[-1] for row in matrix[1:]])
        self.assertEqual(self.namespace["accuracy"], 1.0)
        self.assertEqual(self.outputs[-1], "Accuracy: 100.00%\n")


if __name__ == "__main__":
    unittest.main()
