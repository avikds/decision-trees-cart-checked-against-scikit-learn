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

# Step 3 - grow_tree (not yet solved)
# TODO: implement

# Step 4 - predict_tree (not yet solved)
# TODO: implement

# Step 5 - fit_sklearn_tree (not yet solved)
# TODO: implement

# Step 6 - sklearn_splits (not yet solved)
# TODO: implement

# Step 7 - compare_trees (not yet solved)
# TODO: implement

# Step 8 - moons_data (not yet solved)
# TODO: implement

# Step 9 - overfit_vs_regularized (not yet solved)
# TODO: implement

# Step 10 - rotation_sensitivity (not yet solved)
# TODO: implement

# Step 11 - regression_tree (not yet solved)
# TODO: implement

# Step 12 - tree_rules (not yet solved)
# TODO: implement

# Step 13 - save_and_reload_tree (not yet solved)
# TODO: implement

# Step 14 - predict_species (not yet solved)
# TODO: implement

