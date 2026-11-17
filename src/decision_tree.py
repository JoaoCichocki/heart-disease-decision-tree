"""Decision Tree classifier implemented from scratch with NumPy."""

from collections import Counter
import numpy as np


class TreeNode:
    """A node in the decision tree."""

    def __init__(self, prediction, feature_idx=None, threshold=None,
                 information_gain=0.0, left=None, right=None):
        self.prediction = np.asarray(prediction, dtype=float)
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.information_gain = float(information_gain)
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.feature_idx is None


class DecisionTree:
    """
    Binary/multiclass decision tree using entropy and information gain.

    The implementation is intentionally educational: candidate thresholds
    are selected from the 25th, 50th and 75th percentiles of each feature.
    """

    def __init__(self, max_depth=4, min_samples_leaf=2,
                 min_information_gain=0.0):
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.classes_ = None
        self.tree_ = None
        self.feature_importances_ = None

    @staticmethod
    def _entropy(labels):
        if len(labels) == 0:
            return 0.0
        counts = np.array(list(Counter(labels).values()), dtype=float)
        probabilities = counts / counts.sum()
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _class_probabilities(self, labels):
        counts = np.array([(labels == label).sum() for label in self.classes_],
                          dtype=float)
        return counts / counts.sum()

    def _best_split(self, X, y):
        best_gain = -1.0
        best = None
        parent_entropy = self._entropy(y)

        for feature_idx in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])
            for threshold in np.unique(thresholds):
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask
                left_count = int(left_mask.sum())
                right_count = int(right_mask.sum())

                if (left_count < self.min_samples_leaf or
                        right_count < self.min_samples_leaf):
                    continue

                weighted_entropy = (
                    left_count / len(y) * self._entropy(y[left_mask]) +
                    right_count / len(y) * self._entropy(y[right_mask])
                )
                gain = parent_entropy - weighted_entropy

                if gain > best_gain:
                    best_gain = gain
                    best = (feature_idx, float(threshold), left_mask, right_mask)

        if best is None:
            return None
        return best[0], best[1], best[2], best[3], best_gain

    def _build_tree(self, X, y, depth):
        probabilities = self._class_probabilities(y)
        node = TreeNode(probabilities)

        # Stop when the node is pure or a stopping condition is reached.
        if (depth >= self.max_depth or len(np.unique(y)) == 1):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_idx, threshold, left_mask, right_mask, gain = split
        if gain <= self.min_information_gain:
            return node

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.information_gain = gain
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        """Fit the tree using feature matrix X and target vector y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if len(y) == 0:
            raise ValueError("Training data cannot be empty")
        if not np.isfinite(X).all():
            raise ValueError("X contains missing or infinite values")

        self.classes_ = np.unique(y)
        self.tree_ = self._build_tree(X, y, depth=0)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self._accumulate_importance(self.tree_, len(y))
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _accumulate_importance(self, node, total_samples):
        if node is None or node.is_leaf:
            return
        self.feature_importances_[node.feature_idx] += (
            node.information_gain
        )
        self._accumulate_importance(node.left, total_samples)
        self._accumulate_importance(node.right, total_samples)

    def _predict_one_proba(self, row):
        node = self.tree_
        while not node.is_leaf:
            if row[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict_proba(self, X):
        """Return class probabilities for each row in X."""
        if self.tree_ is None:
            raise RuntimeError("Call fit before predict_proba")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        return np.vstack([self._predict_one_proba(row) for row in X])

    def predict(self, X):
        """Return the most likely class for each row in X."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def print_tree(self, feature_names=None):
        """Print the learned tree in a readable, indented format."""
        if self.tree_ is None:
            raise RuntimeError("Call fit before print_tree")

        def visit(node, depth):
            if node.is_leaf:
                print("    " * depth + f"Leaf: probabilities={np.round(node.prediction, 3)}")
                return
            name = (feature_names[node.feature_idx]
                    if feature_names else f"X[{node.feature_idx}]")
            print("    " * depth + f"{name} < {node.threshold:.3f}")
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)

        visit(self.tree_, 0)
