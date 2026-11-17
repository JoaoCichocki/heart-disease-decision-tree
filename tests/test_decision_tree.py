import unittest
import numpy as np

from src.decision_tree import DecisionTree


class DecisionTreeTests(unittest.TestCase):
    def test_learns_simple_binary_pattern(self):
        X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        model = DecisionTree(max_depth=2, min_samples_leaf=1)
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_probabilities_sum_to_one(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        model = DecisionTree(max_depth=2, min_samples_leaf=1).fit(X, y)
        probabilities = model.predict_proba(X)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_before_fit_raises_error(self):
        model = DecisionTree()
        with self.assertRaises(RuntimeError):
            model.predict([[1.0]])


if __name__ == "__main__":
    unittest.main()
