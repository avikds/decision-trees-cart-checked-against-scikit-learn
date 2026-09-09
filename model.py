"""
Decision Trees: CART, Checked Against Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gini
def gini(y):
    # Gini impurity: 1 - sum(p_k^2)
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    return float(1.0 - np.sum(probabilities ** 2))

# Step 2 - best_split
def best_split(X, y):
    # Calculate the impurity of the parent node.
    n = len(y)
    if n == 0:
        return (None, None, 0.0)

    parent_gini = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    # Scan features in order.
    for j in range(X.shape[1]):
        values = np.sort(np.unique(X[:, j]))

        # No possible split if the feature has fewer than two distinct values.
        if len(values) < 2:
            continue

        # Thresholds are midpoints between consecutive distinct values.
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0

            left_mask = X[:, j] <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            n_left = len(y_left)
            n_right = len(y_right)

            weighted_gini = (
                n_left * gini(y_left) + n_right * gini(y_right)
            ) / n

            gain = parent_gini - weighted_gini

            # Strictly greater preserves the first split on ties.
            if gain > best_gain:
                best_gain = float(gain)
                best_feature = j
                best_threshold = float(threshold)

    # Return no split if nothing improves impurity.
    if best_feature is None:
        return (None, None, 0.0)

    return (best_feature, best_threshold, best_gain)

# Step 3 - grow_tree
def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    n = len(y)

    # Leaf helper: majority class, with ties going to the smallest class.
    def make_leaf():
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        return {
            "leaf": True,
            "value": int(majority_class),
            "n": n
        }

    # Stop if the node is empty, pure, or the maximum depth is reached.
    if n == 0:
        return {
            "leaf": True,
            "value": 0,
            "n": 0
        }

    if len(np.unique(y)) == 1 or depth >= max_depth:
        return make_leaf()

    parent_gini = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    # Search all valid CART splits while enforcing min_samples_leaf.
    for j in range(X.shape[1]):
        values = np.sort(np.unique(X[:, j]))

        if len(values) < 2:
            continue

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0

            left_mask = X[:, j] <= threshold
            right_mask = ~left_mask

            n_left = int(np.sum(left_mask))
            n_right = int(np.sum(right_mask))

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            weighted_gini = (
                n_left * gini(y_left) + n_right * gini(y_right)
            ) / n

            gain = parent_gini - weighted_gini

            # Strict comparison keeps the first best split on ties.
            if gain > best_gain:
                best_gain = float(gain)
                best_feature = j
                best_threshold = float(threshold)

    # No valid split that improves impurity.
    if best_feature is None or best_gain <= 0.0:
        return make_leaf()

    # Partition the data using the selected split.
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_subtree = grow_tree(
        X[left_mask],
        y[left_mask],
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        depth=depth + 1
    )

    right_subtree = grow_tree(
        X[right_mask],
        y[right_mask],
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        depth=depth + 1
    )

    return {
        "leaf": False,
        "feature": best_feature,
        "threshold": best_threshold,
        "n": n,
        "left": left_subtree,
        "right": right_subtree
    }

# Step 4 - predict_tree
def predict_tree(tree, X):
    predictions = []

    for row in X:
        node = tree

        while not node["leaf"]:
            feature = node["feature"]
            threshold = node["threshold"]

            if row[feature] <= threshold:
                node = node["left"]
            else:
                node = node["right"]

        predictions.append(node["value"])

    return np.asarray(predictions, dtype=int)

# Step 5 - fit_sklearn_tree
from sklearn.tree import DecisionTreeClassifier

def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    clf.fit(X, y)

    return clf

# Step 6 - sklearn_splits
def sklearn_splits(clf):
    tree = clf.tree_
    splits = []

    for node in range(tree.node_count):
        # A node is a leaf when children_left is -1.
        if tree.children_left[node] != -1:
            splits.append(
                (
                    int(tree.feature[node]),
                    round(float(tree.threshold[node]), 3)
                )
            )

    return splits

# Step 7 - compare_trees
def compare_trees(X, y, max_depth=2):
    # Fit the hand-built CART tree.
    my_tree = grow_tree(
        X,
        y,
        max_depth=max_depth,
        min_samples_leaf=1
    )

    # Fit the scikit-learn CART tree using the default seed.
    clf = fit_sklearn_tree(
        X,
        y,
        max_depth=max_depth
    )

    # Extract splits from the hand-built tree in depth-first, left-first order.
    my_splits = []

    def collect_splits(node):
        if node["leaf"]:
            return

        my_splits.append(
            (
                int(node["feature"]),
                round(float(node["threshold"]), 3)
            )
        )

        collect_splits(node["left"])
        collect_splits(node["right"])

    collect_splits(my_tree)

    # Get the scikit-learn splits.
    sk_splits = sklearn_splits(clf)

    # Compare predictions on the original data.
    my_predictions = predict_tree(my_tree, X)
    sklearn_predictions = clf.predict(X)

    agreement = float(np.mean(my_predictions == sklearn_predictions))

    return {
        "same_splits": my_splits == sk_splits,
        "agreement": agreement,
        "my_splits": my_splits
    }

# Step 8 - moons_data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

# Step 9 - overfit_vs_regularized
from sklearn.metrics import accuracy_score

def overfit_vs_regularized(
    X_train,
    X_test,
    y_train,
    y_test,
    min_samples_leaf=5
):
    # Unrestricted tree.
    free_tree = fit_sklearn_tree(
        X_train,
        y_train,
        max_depth=None,
        min_samples_leaf=1
    )

    # Regularized tree.
    regularized_tree = fit_sklearn_tree(
        X_train,
        y_train,
        max_depth=None,
        min_samples_leaf=min_samples_leaf
    )

    # Predictions for the unrestricted tree.
    free_train_pred = free_tree.predict(X_train)
    free_test_pred = free_tree.predict(X_test)

    # Predictions for the regularized tree.
    regularized_train_pred = regularized_tree.predict(X_train)
    regularized_test_pred = regularized_tree.predict(X_test)

    return {
        "free": {
            "train_acc": float(accuracy_score(y_train, free_train_pred)),
            "test_acc": float(accuracy_score(y_test, free_test_pred)),
            "leaves": int(free_tree.get_n_leaves())
        },
        "regularized": {
            "train_acc": float(
                accuracy_score(y_train, regularized_train_pred)
            ),
            "test_acc": float(
                accuracy_score(y_test, regularized_test_pred)
            ),
            "leaves": int(regularized_tree.get_n_leaves())
        }
    }

# Step 10 - rotation_sensitivity
def rotation_sensitivity(
    X_train,
    X_test,
    y_train,
    y_test,
    degrees=45.0,
    min_samples_leaf=5
):
    # Convert the rotation angle from degrees to radians.
    theta = np.deg2rad(degrees)

    # 2-D rotation matrix.
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Apply the rotation as X @ R.T.
    X_train_rotated = X_train @ rotation_matrix.T
    X_test_rotated = X_test @ rotation_matrix.T

    # Fit the tree on the original features.
    original_tree = fit_sklearn_tree(
        X_train,
        y_train,
        max_depth=None,
        min_samples_leaf=min_samples_leaf
    )

    # Fit the tree on the rotated features.
    rotated_tree = fit_sklearn_tree(
        X_train_rotated,
        y_train,
        max_depth=None,
        min_samples_leaf=min_samples_leaf
    )

    # Calculate test accuracies.
    original_acc = float(
        accuracy_score(y_test, original_tree.predict(X_test))
    )

    rotated_acc = float(
        accuracy_score(y_test, rotated_tree.predict(X_test_rotated))
    )

    return {
        "original_acc": original_acc,
        "rotated_acc": rotated_acc,
        "drop": round(original_acc - rotated_acc, 4)
    }

# Step 11 - regression_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def regression_tree(X, y, max_depth=2, random_state=42):
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X, y)

    predictions = model.predict(X)

    n_distinct_predictions = int(np.unique(predictions).size)
    train_mse = float(mean_squared_error(y, predictions))

    return {
        "model": model,
        "n_distinct_predictions": n_distinct_predictions,
        "train_mse": train_mse
    }

# Step 12 - tree_rules
from sklearn.tree import export_text

def tree_rules(clf, feature_names):
    return export_text(
        clf,
        feature_names=list(feature_names)
    )

# Step 13 - save_and_reload_tree
import joblib

def save_and_reload_tree(clf, path):
    joblib.dump(clf, path)
    return joblib.load(path)

# Step 14 - predict_species (not yet solved)
# TODO: implement

