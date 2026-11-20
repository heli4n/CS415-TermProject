mport unittest

from src.eval.classification import (
    calculate_metrics,
    bootstrap_confidence_interval
)


class TestClassification(unittest.TestCase):

    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]

        result = calculate_metrics(y_true, y_pred)

        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertEqual(
            result["per_class_recall"],
            {0: 1.0, 1: 1.0, 2: 1.0}
        )

    def test_partial_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]

        result = calculate_metrics(y_true, y_pred)

        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(
            result["confusion_matrix"],
            [[2, 0, 0], [0, 1, 1], [0, 1, 1]]
        )

    def test_bootstrap_reproducibility(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]

        result1 = bootstrap_confidence_interval(
            y_true, y_pred, seed=42
        )
        result2 = bootstrap_confidence_interval(
            y_true, y_pred, seed=42
        )

        self.assertEqual(result1, result2)

    def test_bootstrap_macro_f1(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]

        result = bootstrap_confidence_interval(
            y_true, y_pred, metric="macro_f1"
        )

        self.assertLessEqual(result["lower"], result["upper"])
        self.assertGreaterEqual(result["lower"], 0.0)
        self.assertLessEqual(result["upper"], 1.0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            calculate_metrics([0, 1], [0])

        with self.assertRaises(ValueError):
            calculate_metrics([], [])

        with self.assertRaises(ValueError):
            bootstrap_confidence_interval(
                [0, 1], [0, 1], metric="invalid"
            )


if __name__ == "__main__":
    unittest.main()

