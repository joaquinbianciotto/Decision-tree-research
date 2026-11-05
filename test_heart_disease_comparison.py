"""Regresiones del criterio de selección en el notebook integrado."""

import json
import math
from pathlib import Path
import unittest


NOTEBOOK = Path(__file__).with_name("heart-disease-comparison.ipynb")


class IntegratedDecisionTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        cls.namespace = {"log2": math.log2}
        for cell_id in ("dt-helpers", "dt-support", "dt-training"):
            source = "".join(cells[cell_id]["source"])
            exec(compile(source, cell_id, "exec"), cls.namespace)

    def train(self, *args, **kwargs):
        return self.namespace["entrenar_arbol_enriquecido"](*args, **kwargs)

    def test_gain_ratio_is_used_in_descendants(self):
        rows = (
            [[0, 0, 0, "no"] for _ in range(100)]
            + [[1, 0, 0, "yes"] for _ in range(30)]
            + [[1, 0, 0, "no"] for _ in range(20)]
            + [[1, 1, 0, "yes"] for _ in range(20)]
            + [[1, 1, 0, "no"] for _ in range(30)]
        )
        rows[100][2] = 1
        gain = self.train(rows, [0, 1, 2], criterio="gain", podar=False)
        ratio = self.train(rows, [0, 1, 2], criterio="gain_ratio", podar=False)
        self.assertEqual((gain.attribute, ratio.attribute), (0, 0))
        self.assertEqual(gain.children[1].attribute, 1)
        self.assertEqual(ratio.children[1].attribute, 2)

    def test_gain_ratio_selects_between_numeric_thresholds(self):
        rows = [[float(index), int(label)] for index, label in enumerate("00101111")]
        gain = self.train(rows, [0], continuos=[0], criterio="gain", podar=False)
        ratio = self.train(rows, [0], continuos=[0], criterio="gain_ratio", podar=False)
        self.assertEqual(gain.threshold, 3)
        self.assertEqual(ratio.threshold, 1)

    def test_minimum_gain_applies_to_either_criterion(self):
        rows = [[1, 0], [2, 0], [4, 1], [5, 1]]
        for criterion in ("gain", "gain_ratio"):
            with self.subTest(criterion=criterion):
                tree = self.train(
                    rows, [0], continuos=[0], criterio=criterion,
                    min_gain=2, podar=False,
                )
                self.assertIsNone(tree.attribute)

    def test_unknown_criterion_is_rejected(self):
        with self.assertRaises(ValueError):
            self.train([[0, "no"], [1, "yes"]], [0], criterio="unknown")


if __name__ == "__main__":
    unittest.main()
