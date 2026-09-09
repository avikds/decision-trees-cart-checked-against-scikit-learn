"""
Decision Trees: CART, Checked Against Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Decision trees: CART by hand, checked against scikit-learn (Hands-On ML, chapter 5).

Story: grow a Gini tree on iris and confirm it matches DecisionTreeClassifier split
for split; watch a free tree memorize the moons data and min_samples_leaf fix it;
measure the axis-alignment bias with a rotation; fit a regression tree and count
its plateaus; then export the rules, save the model and serve species names.
"""
import os
import tempfile
import numpy as np
from sklearn.datasets import load_iris


def main() -> None:
    iris = load_iris()
    X2, y = iris.data[:, 2:], iris.target

    # ---- 1. Hand-grown CART vs scikit-learn ----
    print(f"iris root impurity {gini(y):.4f}; best split {best_split(X2, y)}")
    cmp = compare_trees(X2, y, max_depth=2)
    print(f"my depth-2 splits {cmp['my_splits']} | same as scikit-learn: {cmp['same_splits']} | "
          f"prediction agreement {cmp['agreement']:.3f}")
    print(tree_rules(fit_sklearn_tree(X2, y), iris.feature_names[2:]))

    # ---- 2. Overfitting and regularization on moons ----
    Xtr, Xte, ytr, yte = moons_data()
    r = overfit_vs_regularized(Xtr, Xte, ytr, yte, min_samples_leaf=5)
    print(f"moons free tree:        train {r['free']['train_acc']:.3f}  test {r['free']['test_acc']:.3f}  leaves {r['free']['leaves']}")
    print(f"moons min_samples_leaf=5: train {r['regularized']['train_acc']:.3f}  test {r['regularized']['test_acc']:.3f}  leaves {r['regularized']['leaves']}")
    rot = rotation_sensitivity(Xtr, Xte, ytr, yte, degrees=45.0)
    print(f"rotate the features 45 degrees: test accuracy {rot['original_acc']:.3f} -> {rot['rotated_acc']:.3f} (drop {rot['drop']:+.3f})")

    # ---- 3. Regression tree ----
    Xq = np.linspace(-1, 1, 200).reshape(-1, 1)
    yq = Xq.ravel() ** 2
    for d in (2, 3, 5):
        rr = regression_tree(Xq, yq, max_depth=d)
        print(f"regression tree depth {d}: {rr['n_distinct_predictions']:>2} distinct predictions, train MSE {rr['train_mse']:.4f}")

    # ---- 4. Ship ----
    final = fit_sklearn_tree(iris.data, iris.target, max_depth=3)
    path = os.path.join(tempfile.gettempdir(), "iris_tree.pkl")
    served = save_and_reload_tree(final, path)
    flowers = [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 4.2, 1.5], [6.7, 3.0, 5.2, 2.3]]
    print(f"\nserved: {predict_species(served, flowers, iris.target_names)} for {flowers}")


if __name__ == "__main__":
    main()

