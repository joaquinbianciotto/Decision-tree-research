"""Execute and check the standalone Random Forest notebook without running DT."""

import ast
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).parent


def original_dataset():
    """Read literal rows only; never execute the decision-tree notebook."""
    notebook = json.loads((ROOT / "decision-tree.ipynb").read_text(encoding="utf-8"))
    matrices = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = getattr(node.targets[0], "id", None)
            if name == "training_matrix":
                matrices[name] = ast.literal_eval(node.value)
            elif name == "test_matrix":
                # The first element is columns.copy(); extract only literal rows.
                matrices[name] = [matrices["training_matrix"][0].copy()] + [
                    ast.literal_eval(row) for row in node.value.elts[1:]
                ]
    return matrices


def execute_notebook():
    notebook = json.loads((ROOT / "random-forest.ipynb").read_text(encoding="utf-8"))
    namespace, output, fits, encoder_fits = {}, io.StringIO(), [], []
    pipeline_fit, encoder_fit = Pipeline.fit, OneHotEncoder.fit
    path_open = Path.open

    def guard_original_notebook(self, *args, **kwargs):
        if self.resolve() == (ROOT / "decision-tree.ipynb").resolve():
            raise AssertionError("The Random Forest notebook must not read or run DT")
        return path_open(self, *args, **kwargs)

    def record_pipeline_fit(self, X, y=None, **kwargs):
        fits.append((deepcopy(X), deepcopy(y)))
        return pipeline_fit(self, X, y, **kwargs)

    def record_encoder_fit(self, X, y=None):
        encoder_fits.append(deepcopy(X))
        return encoder_fit(self, X, y)

    with (patch.object(Pipeline, "fit", record_pipeline_fit),
          patch.object(OneHotEncoder, "fit", record_encoder_fit),
          patch.object(Path, "open", guard_original_notebook),
          contextlib.redirect_stdout(output)):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"rf-cell-{index}", "exec"), namespace)
    return notebook, namespace, output.getvalue(), fits, encoder_fits


class RandomForestNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.ns, cls.output, cls.fits, cls.encoder_fits = execute_notebook()

    def test_dataset_matches_working_notebook_without_executing_it(self):
        for name, matrix in original_dataset().items():
            self.assertEqual(self.ns[name], matrix)
        self.assertEqual(self.ns["columns"][-1], "Play")
        self.assertEqual(len(self.ns["X_train"]), 14)
        self.assertEqual(len(self.ns["X_test"]), 6)

    def test_target_is_separate_from_features(self):
        for split, matrix in (("train", "training_matrix"), ("test", "test_matrix")):
            rows = self.ns[matrix][1:]
            self.assertEqual(self.ns[f"X_{split}"], [row[:-1] for row in rows])
            self.assertEqual(self.ns[f"y_{split}"], [row[-1] for row in rows])
            self.assertTrue(all(len(row) == 4 for row in self.ns[f"X_{split}"]))

    def test_pipeline_configuration(self):
        model = self.ns["model"]
        self.assertIsInstance(model, Pipeline)
        self.assertEqual(len(model.steps), 2)
        encoder, forest = (step for _, step in model.steps)
        self.assertIsInstance(encoder, OneHotEncoder)
        self.assertEqual(encoder.handle_unknown, "ignore")
        self.assertIsInstance(forest, RandomForestClassifier)
        self.assertEqual(forest.n_estimators, 100)
        self.assertEqual(forest.random_state, 42)

    def test_pipeline_and_encoder_fit_only_training_rows_once(self):
        self.assertEqual(self.fits, [(self.ns["X_train"], self.ns["y_train"])])
        self.assertEqual(self.encoder_fits, [self.ns["X_train"]])

    def test_six_valid_predictions_and_labeled_output(self):
        predictions = self.ns["predictions"].tolist()
        self.assertEqual(len(predictions), 6)
        self.assertTrue(set(predictions) <= set(self.ns["y_train"]))
        lines = self.output.splitlines()
        self.assertEqual(len(lines), 6)
        for row, prediction, line in zip(self.ns["X_test"], predictions, lines):
            self.assertEqual(line, f"{dict(zip(self.ns['columns'][:-1], row))}"
                                  f" -> Predicted Play: {prediction}")

    def test_predictions_are_reproducible_from_fresh_execution(self):
        _, namespace, _, _, _ = execute_notebook()
        self.assertEqual(self.ns["predictions"].tolist(), namespace["predictions"].tolist())

    def test_unseen_category_is_accepted_without_refitting(self):
        row = self.ns["X_test"][0].copy()
        row[0] = "snow"
        encoder = self.ns["model"].steps[0][1]
        before = [values.tolist() for values in encoder.categories_]
        prediction = self.ns["model"].predict([row])
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], self.ns["y_train"])
        self.assertEqual(before, [values.tolist() for values in encoder.categories_])


if __name__ == "__main__":
    unittest.main()
