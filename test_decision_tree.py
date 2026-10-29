"""Test the recursive decision tree under both documented criterion settings.

Run with: python -m unittest -v test_decision_tree.py
"""

import ast
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch


HAS_MATPLOTLIB = importlib.util.find_spec('matplotlib') is not None
if HAS_MATPLOTLIB:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


NOTEBOOK = Path(__file__).with_name("decision-tree.ipynb")
SCORE_LINES = (
    "            score = gain  # Information Gain\n",
    "            # score = gain / split_info if split_info else 0.0  # Gain Ratio\n",
)


def raw_gain_tradeoff_rows():
    """A rare branch wins Gain Ratio despite having less raw gain."""
    rows = (
        [["left", "common", "yes"] for _ in range(30)]
        + [["left", "common", "no"] for _ in range(20)]
        + [["right", "common", "yes"] for _ in range(20)]
        + [["right", "common", "no"] for _ in range(30)]
    )
    rows[0][1] = "rare"
    return rows


def root_weighting_rows():
    """A 10-row mixed child has local gain 1 within 100 original rows."""
    return (
        [["large", "a", "no"] for _ in range(45)]
        + [["large", "b", "no"] for _ in range(45)]
        + [["small", "a", "yes"] for _ in range(5)]
        + [["small", "b", "no"] for _ in range(5)]
    )


def parity_rows():
    """Three categorical bits require three levels when zero gain is allowed."""
    return [[a, b, c, "yes" if (a + b + c) % 2 else "no"]
            for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def tree_depth(tree):
    if tree.attribute is None:
        return 0
    return 1 + max(tree_depth(child) for child in tree.children.values())


class DecisionTreeNotebookTests(unittest.TestCase):
    use_gain_ratio = False

    def setUp(self):
        if HAS_MATPLOTLIB:
            show = patch.object(plt, 'show')
            self.show = show.start()
            self.addCleanup(show.stop)
            self.addCleanup(plt.close, 'all')
        self.original_source = NOTEBOOK.read_text(encoding="utf-8")
        self.assertTrue(self.original_source.strip(), "The notebook must contain valid notebook JSON.")
        self.notebook = json.loads(self.original_source)
        self.cells = [
            dict(cell, source=cell["source"].copy())
            for cell in self.notebook["cells"] if cell["cell_type"] == "code"
        ]
        training_source = "".join(self.cells[3]["source"])
        pair = "".join(SCORE_LINES)
        self.assertEqual(training_source.count(pair), 1)
        if self.use_gain_ratio:
            switched = (SCORE_LINES[0].replace("score =", "# score =", 1)
                        + SCORE_LINES[1].replace("# score =", "score =", 1))
            self.cells[3]["source"] = training_source.replace(pair, switched, 1).splitlines(keepends=True)
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
            if index == 1:
                self.assertEqual(imports, [])
            for node in imports:
                modules = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                allowed = {"matplotlib.pyplot"} if index == 3 else {"collections", "math"}
                self.assertTrue(set(modules) <= allowed)

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
        self.assertIsNone(node.sample_count)
        self.assertIsNone(node.predicted_class_count)
        branch = self.namespace['Node']('yes', 0, 1.5)
        self.assertEqual((branch.attribute, branch.threshold), (0, 1.5))

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

    def test_split_information_uses_branch_proportions(self):
        split_info = self.namespace["split_information"]
        cases = [
            ([], 0.0),
            ([["a", "yes"], ["a", "no"]], 0.0),
            ([["a", "yes"], ["b", "yes"]], 1.0),
            ([[i, "yes"] for i in range(4)], 2.0),
            ([["a", "yes"]] * 3 + [["b", "yes"]], 0.8112781244591328),
        ]
        for rows, expected in cases:
            with self.subTest(rows=rows):
                self.assertAlmostEqual(split_info(0, rows), expected)

    def test_criterion_switch_does_not_change_notebook_on_disk(self):
        self.assertEqual(NOTEBOOK.read_text(encoding="utf-8"), self.original_source)
        active = SCORE_LINES[1].replace("# score =", "score =", 1) if self.use_gain_ratio else SCORE_LINES[0]
        self.assertIn(active, self.cells[3]["source"])

    def test_training_uses_direct_recursion_without_nested_helpers(self):
        module = ast.parse("".join(self.cells[3]["source"]))
        training = next(node for node in module.body
                         if isinstance(node, ast.FunctionDef) and node.name == "decision_tree")
        self.assertEqual([argument.arg for argument in training.args.args],
                         ["D", "A", "min_gain", "max_depth", "min_samples_split", "depth", "n_total"])
        self.assertEqual([ast.literal_eval(value) for value in training.args.defaults],
                         [1e-12, None, 2, 0, None])
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
        self.assertEqual([(keyword.arg, ast.unparse(keyword.value))
                          for keyword in calls[0].keywords],
                         [("max_depth", "max_depth"),
                          ("min_samples_split", "min_samples_split"),
                          ("depth", "depth + 1"),
                          ("n_total", "n_total")])
        helpers = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"predict_one", "print_tree"} <= helpers)
        example = next(node.value for node in module.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "tree"
                               for target in node.targets))
        self.assertEqual([argument.id for argument in example.args], ["D", "A"])
        self.assertEqual({keyword.arg: ast.literal_eval(keyword.value)
                          for keyword in example.keywords},
                         {"max_depth": None, "min_samples_split": 2})

    def test_criterion_switch_changes_high_cardinality_choice(self):
        rows = [
            [f"id-{i}", "low" if i < 4 else "high", "no" if i < 4 else "yes"]
            for i in range(8)
        ]
        tree = self.train(rows, [0, 1])
        expected_attribute = 1 if self.use_gain_ratio else 0
        self.assertEqual(tree.attribute, expected_attribute)
        self.assertEqual(set(tree.children), {row[expected_attribute] for row in rows})
        self.assertTrue(all(child.attribute is None for child in tree.children.values()))
        self.assertEqual(
            [self.namespace["predict_one"](tree, row[:-1]) for row in rows],
            [row[-1] for row in rows],
        )
        self.assertEqual(self.train(rows, [1, 0]).attribute, 1)

    def test_active_score_can_rank_differently_from_raw_gain(self):
        rows = raw_gain_tradeoff_rows()
        tree = self.train(rows, [0, 1])
        self.assertEqual(tree.attribute, 1 if self.use_gain_ratio else 0)
        self.assertEqual(set(tree.children), {"common", "rare"} if self.use_gain_ratio
                         else {"left", "right"})

    def test_equal_scores_keep_first_attribute_order(self):
        rows = [["a", "x", "no"], ["b", "y", "yes"]]
        for attributes in ([0, 1], [1, 0]):
            with self.subTest(attributes=attributes):
                self.assertEqual(self.train(rows, attributes).attribute, attributes[0])

    def test_minimum_uses_raw_gain_when_ratio_is_smaller(self):
        rows = [[i, "no" if i < 4 else "yes"] for i in range(8)]
        tree = self.train(rows, [0], min_gain=1.0)
        # Raw gain is 1 bit; Gain Ratio is 1/3, below the same threshold.
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(set(tree.children), set(range(8)))

    def test_minimum_uses_raw_gain_when_ratio_is_larger(self):
        rows = [["common", "no"] for _ in range(7)] + [["rare", "yes"]]
        tree = self.train(rows, [0], min_gain=0.75)
        # Raw gain is about 0.544 bits, below the threshold; Gain Ratio is 1.
        self.assertIsNone(tree.attribute)
        self.assertEqual(tree.prediction, "no")
        self.assertEqual(tree.children, {})

    def test_minimum_applies_to_selected_attribute_without_substitution(self):
        tree = self.train(raw_gain_tradeoff_rows(), [0, 1], min_gain=0.02)
        # Attribute 0 has raw gain about 0.029; attribute 1 has about 0.010.
        if self.use_gain_ratio:
            self.assertIsNone(tree.attribute)
            self.assertEqual(tree.prediction, "yes")
            self.assertEqual(tree.children, {})
        else:
            self.assertEqual(tree.attribute, 0)

    def test_minimum_uses_original_training_weight_at_descendant(self):
        rows = root_weighting_rows()
        small_rows = [row for row in rows if row[0] == "small"]
        local_gain = (self.namespace["impurityEval_1"](small_rows)
                      - self.namespace["impurityEval_2"](1, small_rows))
        self.assertEqual(local_gain, 1.0)
        self.assertEqual(len(small_rows) / len(rows) * local_gain, 0.1)
        for minimum, expected_attribute in ((0.15, None), (0.1, 1)):
            with self.subTest(min_gain=minimum):
                tree = self.train(rows, [0, 1], min_gain=minimum)
                self.assertEqual(tree.attribute, 0)
                self.assertEqual(tree.prediction, "no")
                self.assertEqual(tree.children["large"].prediction, "no")
                small = tree.children["small"]
                self.assertEqual(small.attribute, expected_attribute)
                self.assertEqual(small.prediction, "yes")
                if minimum == 0.1:
                    self.assertEqual(small.children["a"].prediction, "yes")
                    self.assertEqual(small.children["b"].prediction, "no")
                else:
                    self.assertEqual(small.children, {})
                    self.assertEqual(self.namespace["predict_one"](tree, ["small", "b"]), "yes")

    def test_root_weight_is_one_with_default_or_explicit_total(self):
        rows = [[i, "no" if i < 4 else "yes"] for i in range(8)]
        for total in (None, len(rows)):
            for minimum, expected in ((1.0, 0), (1.01, None)):
                with self.subTest(n_total=total, min_gain=minimum):
                    tree = self.train(rows, [0], min_gain=minimum, n_total=total)
                    self.assertEqual(tree.attribute, expected)
                    self.assertEqual(tree.prediction, "no")
                    self.assertEqual(len(tree.children), 8 if expected == 0 else 0)

    def test_original_total_is_forwarded_across_multiple_levels(self):
        original = self.namespace["decision_tree"]
        calls = []

        def record(rows, attributes, *args, **options):
            calls.append((len(rows), options.get("depth", 0), options.get("n_total")))
            return original(rows, attributes, *args, **options)

        self.namespace["decision_tree"] = record
        try:
            tree = self.train(parity_rows(), [0, 1, 2], min_gain=0)
        finally:
            self.namespace["decision_tree"] = original
        self.assertEqual(tree_depth(tree), 3)
        self.assertEqual(len(calls), 15)
        self.assertEqual(calls[0], (8, 0, None))
        self.assertEqual({(size, depth) for size, depth, _ in calls[1:]},
                         {(4, 1), (2, 2), (1, 3)})
        self.assertTrue(all(total == 8 for _, _, total in calls[1:]))

    def test_independent_training_calls_reinitialize_original_total(self):
        for count in (8, 2):
            with self.subTest(count=count):
                rows = [[i, "no" if i < count // 2 else "yes"] for i in range(count)]
                tree = self.train(rows, [0], min_gain=0.75)
                self.assertEqual(tree.attribute, 0)
                self.assertEqual(len(tree.children), count)

    def test_original_six_positional_arguments_remain_compatible(self):
        rows = [["a", "no"], ["b", "yes"], ["c", "yes"]]
        cases = [
            ((0, None, 2, 0), 0),
            ((0, 0, 2, 0), None),
            ((0, None, 4, 0), None),
            ((0, 1, 2, 1), None),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                tree = self.namespace["decision_tree"](rows, [0], *arguments)
                self.assertEqual(tree.attribute, expected)
                self.assertEqual(tree.prediction, "yes")

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

    def test_reached_depth_limit_returns_current_majority(self):
        rows = [["a", "no"], ["b", "yes"], ["c", "yes"]]
        for maximum, depth in ((0, 0), (2, 2), (2, 3)):
            with self.subTest(max_depth=maximum, depth=depth):
                tree = self.train(rows, [0], max_depth=maximum, depth=depth)
                self.assertIsNone(tree.attribute)
                self.assertEqual(tree.prediction, "yes")
                self.assertEqual(tree.children, {})
                self.assertEqual(self.namespace["predict_one"](tree, ["unseen"]), "yes")

    def test_depth_limits_bound_tree_height(self):
        rows = parity_rows()
        for maximum, expected in ((0, 0), (1, 1), (2, 2), (3, 3), (None, 3)):
            with self.subTest(max_depth=maximum):
                tree = self.train(rows, [0, 1, 2], min_gain=0, max_depth=maximum)
                self.assertEqual(tree_depth(tree), expected)
                if maximum is None or maximum == 3:
                    self.assertEqual(
                        [self.namespace["predict_one"](tree, row[:-1]) for row in rows],
                        [row[-1] for row in rows],
                    )

    def test_depth_one_uses_descendant_majority_not_root_majority(self):
        tree = self.train(self.namespace["D"], self.namespace["A"], max_depth=1)
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(tree.prediction, "yes")
        self.assertTrue(all(child.attribute is None for child in tree.children.values()))
        self.assertEqual(tree.children["sunny"].prediction, "no")
        self.assertEqual(tree.children["rain"].prediction, "yes")
        self.assertEqual(self.namespace["predict_one"](tree,
                         ["sunny", "cool", "normal", "weak"]), "no")

    def test_sample_minimum_boundary_allows_equality(self):
        rows = [["a", "no"], ["b", "yes"], ["c", "yes"], ["d", "yes"]]
        for minimum, expected_attribute in ((3, 0), (4, 0), (5, None)):
            with self.subTest(min_samples_split=minimum):
                tree = self.train(rows, [0], min_samples_split=minimum)
                self.assertEqual(tree.attribute, expected_attribute)
                self.assertEqual(tree.prediction, "yes")
                self.assertEqual(set(tree.children), set() if minimum == 5
                                 else {"a", "b", "c", "d"})

    def test_sample_minimum_is_node_local_not_a_minimum_leaf_size(self):
        for minimum, expected_sunny_attribute in ((5, 2), (6, None)):
            with self.subTest(min_samples_split=minimum):
                tree = self.train(self.namespace["D"], self.namespace["A"],
                                  min_samples_split=minimum)
                self.assertEqual(tree.attribute, 0)
                sunny = tree.children["sunny"]  # Five rows, unlike the 14-row root.
                self.assertEqual(sunny.attribute, expected_sunny_attribute)
                self.assertEqual(sunny.prediction, "no")
                if minimum == 5:
                    # Equality permits children with only three and two training rows.
                    self.assertEqual(set(sunny.children), {"high", "normal"})
                    self.assertEqual(sunny.children["high"].prediction, "no")
                    self.assertEqual(sunny.children["normal"].prediction, "yes")
                else:
                    self.assertEqual(sunny.children, {})

    def test_depth_and_sample_stops_combine(self):
        for maximum, minimum, expected in ((1, 2, 1), (2, 5, 1), (0, 9, 0), (2, 4, 2)):
            with self.subTest(max_depth=maximum, min_samples_split=minimum):
                tree = self.train(parity_rows(), [0, 1, 2], min_gain=0,
                                  max_depth=maximum, min_samples_split=minimum)
                self.assertEqual(tree_depth(tree), expected)

    def test_original_positional_min_gain_remains_compatible(self):
        rows = [["a", "no"], ["a", "yes"], ["a", "yes"]]
        tree = self.namespace["decision_tree"](rows, [0], 0)
        self.assertEqual(tree.attribute, 0)
        self.assertEqual(tree.children["a"].prediction, "yes")
        self.assertIsNone(tree.children["a"].attribute)

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

    def test_leaf_counts_cover_pure_and_majority_stopping_rules(self):
        examples = [
            ([["a", "yes"], ["b", "yes"]], [0], {}, "yes", 2, 2),
            ([["a", "no"], ["b", "yes"], ["c", "yes"]], [], {}, "yes", 3, 2),
            ([["a", "no"], ["a", "yes"], ["a", "yes"]], [0], {}, "yes", 3, 2),
            ([["a", "no"], ["b", "yes"]], [0], {"min_gain": 2}, "no", 2, 1),
        ]
        for rows, attributes, options, prediction, total, matching in examples:
            with self.subTest(rows=rows, options=options):
                leaf = self.train(rows, attributes, **options)
                self.assertIsNone(leaf.attribute)
                self.assertEqual((leaf.prediction, leaf.sample_count,
                                  leaf.predicted_class_count), (prediction, total, matching))

    def test_recursive_leaf_counts_use_only_reaching_training_rows(self):
        rows = [['a', 'yes'], ['a', 'yes'], ['a', 'no'], ['b', 'no'], ['b', 'no']]
        tree = self.train(rows, [0])
        self.assertEqual(tree.attribute, 0)
        self.assertEqual((tree.children['a'].sample_count,
                          tree.children['a'].predicted_class_count), (3, 2))
        self.assertEqual((tree.children['b'].sample_count,
                          tree.children['b'].predicted_class_count), (2, 2))
        predict = self.namespace['predict_one']
        self.assertEqual([predict(tree, [value]) for value in ('a', 'b', 'unseen')],
                         ['yes', 'no', 'no'])
        self.assertEqual(sum(child.sample_count for child in tree.children.values()), len(rows))
        constant = self.train([['a', 'no'], ['a', 'yes'], ['a', 'yes']], [0], min_gain=0)
        self.assertEqual((constant.children['a'].sample_count,
                          constant.children['a'].predicted_class_count), (3, 2))

    def test_plot_labels_include_leaf_counts_without_rendering(self):
        rows = [['a', 'yes'], ['a', 'yes'], ['a', 'no'], ['b', 'no']]
        tree = self.train(rows, [0])
        pyplot = MagicMock()
        fig, ax = MagicMock(), MagicMock()
        pyplot.subplots.return_value = fig, ax
        with patch.dict('sys.modules', {'matplotlib': MagicMock(pyplot=pyplot),
                                       'matplotlib.pyplot': pyplot}):
            self.namespace['plot_tree'](tree, ['Feature'])
            labels = [call.args[2] for call in ax.text.call_args_list]
            self.assertEqual(labels, [
                'a', 'b',
                'Class: yes\n2/3',
                'Class: no\n1/1',
                'Feature',
            ])
            ax.text.reset_mock()
            self.namespace['plot_tree'](self.namespace['Node']('manual'), [])
            self.assertEqual(ax.text.call_args.args[2],
                             'Class: manual\nunknown/unknown')
        self.assertEqual((tree.children['a'].sample_count,
                          tree.children['a'].predicted_class_count), (3, 2))

    def test_tree_layout_matches_trained_structure_without_mutation(self):
        tree = self.namespace['tree']
        nodes, edges = self.namespace['tree_layout'](tree)
        self.assertEqual(len(nodes), 8)
        self.assertEqual(len(edges), 7)
        positions = {node: position for node, position in nodes}
        self.assertEqual(len(set(positions.values())), 8)
        for node, start in nodes:
            for branch, child in node.children.items():
                self.assertIn((start, positions[child], str(branch)), edges)
                self.assertEqual(positions[child][1], start[1] - 1)
        self.assertEqual([self.namespace['predict_one'](tree, row[:-1])
                          for row in self.namespace['D']],
                         [row[-1] for row in self.namespace['D']])

    def test_tree_layout_single_leaf_and_repeated_multiway_labels(self):
        Node = self.namespace['Node']
        layout = self.namespace['tree_layout']
        leaf = Node('same')
        self.assertEqual(layout(leaf), ([(leaf, (0, 0))], []))
        for count in (1, 3, 12):
            with self.subTest(branch_count=count):
                root = Node('same', attribute=0)
                for branch in range(count):
                    child = Node('same', attribute=0)
                    child.children['repeated'] = Node('same')
                    root.children[branch] = child
                nodes, edges = layout(root)
                self.assertEqual(len(nodes), 1 + 2 * count)
                self.assertEqual(len(edges), 2 * count)
                self.assertEqual(len({position for _, position in nodes}), len(nodes))
                self.assertEqual(sum(label == 'repeated' for _, _, label in edges), count)
                self.assertEqual(nodes[-1][1], ((count - 1) / 2, 0))

    @unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib is not installed; no installation allowed')
    def test_plot_artists_and_same_cell_display(self):
        self.show.assert_called_once_with()
        for tree in (self.namespace['tree'], self.namespace['Node']('single')):
            with self.subTest(tree=tree):
                nodes, edges = self.namespace['tree_layout'](tree)
                fig, ax = self.namespace['plot_tree'](tree, self.namespace['columns'])
                fig.canvas.draw()
                self.assertEqual(len(ax.lines), len(edges))
                self.assertEqual(len(ax.texts), len(nodes) + len(edges))
                self.assertEqual([text.get_text() for text in ax.texts[:len(edges)]],
                                 [label for _, _, label in edges])
                self.assertEqual([text.get_text() for text in ax.texts[len(edges):]],
                                 [(f'Class: {node.prediction}'
                                   f'\n{node.predicted_class_count if node.predicted_class_count is not None else "unknown"}'
                                   f'/{node.sample_count if node.sample_count is not None else "unknown"}')
                                  if node.attribute is None
                                  else self.namespace['columns'][node.attribute]
                                  for node, _ in nodes])
                self.assertFalse(ax.axison)
                plt.close(fig)

    def test_plot_missing_dependency_reports_limitation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with patch('builtins.__import__', side_effect=ModuleNotFoundError(
                    "No module named 'matplotlib'", name='matplotlib')):
                result = self.namespace['plot_tree'](self.namespace['tree'], [])
        self.assertIsNone(result)
        self.assertIn('matplotlib is not installed', output.getvalue())

    def test_final_cell_reports_held_out_accuracy_and_predictions(self):
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
        self.assertEqual(self.outputs[-1],
                         "Accuracy: 100.00%\n['no', 'yes', 'yes', 'yes', 'no', 'yes']\n")


class DecisionTreeGainRatioNotebookTests(DecisionTreeNotebookTests):
    """Run the same regression suite with only the documented line switch."""

    use_gain_ratio = True


if __name__ == "__main__":
    unittest.main()
