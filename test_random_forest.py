"""Execute the standalone RF notebook offline with real pandas and sklearn."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).parent


def synthetic_dataset():
    """Provide 100 complete multiclass rows and two independently missing rows."""
    index = [1000 + 3 * row for row in range(102)]
    features = pd.DataFrame({
        f"feature_{column}": [float((row * (column + 1)) % 107) for row in range(102)]
        for column in range(13)
    }, index=index)
    targets = pd.DataFrame({"num": [row % 5 for row in range(102)]}, index=index)
    features.loc[index[100], "feature_3"] = float("nan")
    targets.loc[index[101], "num"] = float("nan")
    return SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))


def execute_notebook():
    notebook = json.loads((ROOT / "random-forest.ipynb").read_text(encoding="utf-8"))
    namespace, output, fits = {}, io.StringIO(), []
    forest_fit, path_open = RandomForestClassifier.fit, Path.open

    def guard_original_notebook(self, *args, **kwargs):
        if self.resolve() == (ROOT / "decision-tree.ipynb").resolve():
            raise AssertionError("RF must not read or execute the DT notebook")
        return path_open(self, *args, **kwargs)

    def record_fit(self, X, y, **kwargs):
        fits.append((deepcopy(X), deepcopy(y)))
        return forest_fit(self, X, y, **kwargs)

    with (patch("ucimlrepo.fetch_ucirepo", return_value=synthetic_dataset()) as fetch,
          patch.object(RandomForestClassifier, "fit", record_fit),
          patch.object(Path, "open", guard_original_notebook),
          patch("socket.create_connection", side_effect=AssertionError("Offline test")),
          contextlib.redirect_stdout(output)):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"rf-cell-{index}", "exec"), namespace)
        fetch.assert_called_once_with(id=45)
    return notebook, namespace, output.getvalue(), fits


class RandomForestNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.ns, cls.output, cls.fits = execute_notebook()

    def test_clean_notebook_structure_and_stable_ids(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual([cell["id"] for cell in self.notebook["cells"]], [
            "introduction", "data-description", "datasets", "training-description",
            "train-model", "prediction-description", "predictions",
        ])
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_cleanup_and_exact_stratified_split(self):
        data = synthetic_dataset().data
        expected = pd.concat([data.features, data.targets], axis=1)
        expected = expected.rename(columns={"num": "target"}).dropna()
        train, test = train_test_split(
            expected, test_size=0.2, stratify=expected["target"], random_state=55,
        )
        assert_frame_equal(self.ns["df"], expected)
        assert_frame_equal(self.ns["train_df"], train)
        assert_frame_equal(self.ns["test_df"], test)
        self.assertEqual((len(train), len(test)), (80, 20))
        self.assertEqual(set(train["target"]), set(range(5)))
        self.assertEqual(set(test["target"]), set(range(5)))
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertEqual(set(train.index) | set(test.index), set(expected.index))
        self.assertTrue(all(count == 4 for count in test["target"].value_counts()))
        assert_frame_equal(self.ns["heart_disease"].data.features, data.features)
        assert_frame_equal(self.ns["heart_disease"].data.targets, data.targets)

    def test_matches_current_dt_preparation_offline(self):
        # Execute only DT's data cell, never its algorithm, plots, or evaluation.
        notebook = json.loads((ROOT / "decision-tree.ipynb").read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "training-data")
        namespace = {}
        with (patch("ucimlrepo.fetch_ucirepo", return_value=synthetic_dataset()) as fetch,
              patch("socket.create_connection", side_effect=AssertionError("Offline test"))):
            exec(compile("".join(cell["source"]), "dt-data-only", "exec"), namespace)
            fetch.assert_called_once_with(id=45)
        for name in ("df", "train_df", "test_df"):
            assert_frame_equal(self.ns[name], namespace[name])
        for name in ("columns", "training_matrix", "test_matrix"):
            self.assertEqual(self.ns[name], namespace[name])

    def test_target_is_last_and_separate_from_numeric_features(self):
        self.assertEqual(self.ns["columns"][-1], "target")
        self.assertNotIn("num", self.ns["columns"])
        for split, matrix in (("train", "training_matrix"), ("test", "test_matrix")):
            rows = self.ns[matrix][1:]
            self.assertEqual(self.ns[matrix][0], self.ns["columns"])
            self.assertEqual(self.ns[f"X_{split}"], [row[:-1] for row in rows])
            self.assertEqual(self.ns[f"y_{split}"], [row[-1] for row in rows])
            self.assertTrue(all(len(row) == 13 for row in self.ns[f"X_{split}"]))
            self.assertTrue(all(isinstance(value, (int, float))
                                for row in self.ns[f"X_{split}"] for value in row))

    def test_numeric_forest_configuration_and_training_only_fit(self):
        model = self.ns["model"]
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(len(model.estimators_), 100)
        self.assertEqual(model.n_features_in_, 13)
        self.assertEqual(self.fits, [(self.ns["X_train"], self.ns["y_train"])])

    def test_held_out_predictions_and_labeled_output(self):
        predictions = self.ns["predictions"].tolist()
        self.assertEqual(len(predictions), len(self.ns["X_test"]))
        self.assertTrue(set(predictions) <= set(range(5)))
        lines = self.output.splitlines()
        self.assertEqual(len(lines), len(predictions))
        for row, prediction, line in zip(self.ns["X_test"], predictions, lines):
            self.assertEqual(line, f"{dict(zip(self.ns['columns'][:-1], row))}"
                                  f" -> Predicted target: {prediction}")

    def test_predictions_are_reproducible_from_fresh_execution(self):
        _, namespace, output, fits = execute_notebook()
        self.assertEqual(self.ns["predictions"].tolist(), namespace["predictions"].tolist())
        self.assertEqual(self.output, output)
        self.assertEqual(self.fits, fits)


if __name__ == "__main__":
    unittest.main()
