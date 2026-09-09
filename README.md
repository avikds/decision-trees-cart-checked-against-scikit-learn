# Decision Trees: CART, Checked Against Scikit-Learn

Chapter 5 of Hands-On Machine Learning, done the way you would verify your understanding on the job: grow a CART classifier by hand with Gini impurity, then check it split for split against scikit-learn's DecisionTreeClassifier on the iris data; see overfitting appear and disappear with min_samples_leaf on the moons data; watch a tree's sensitivity to rotated features; fit a regression tree and see its piecewise-constant predictions; then save the scikit-learn model and serve it on raw flower measurements.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** gini
- [x] **2.** best_split
- [x] **3.** grow_tree
- [x] **4.** predict_tree
- [x] **5.** fit_sklearn_tree
- [x] **6.** sklearn_splits
- [x] **7.** compare_trees
- [x] **8.** moons_data
- [x] **9.** overfit_vs_regularized
- [x] **10.** rotation_sensitivity
- [x] **11.** regression_tree
- [x] **12.** tree_rules
- [x] **13.** save_and_reload_tree
- [x] **14.** predict_species

## Results

```
iris root impurity 0.6667; best split (0, 2.45, 0.3333333333333334)
my depth-2 splits [(0, 2.45), (1, 1.75)] | same as scikit-learn: True | prediction agreement 1.000
|--- petal length (cm) <= 2.45
|   |--- class: 0
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.75
|   |   |--- class: 1
|   |--- petal width (cm) >  1.75
|   |   |--- class: 2

moons free tree:        train 1.000  test 0.922  leaves 24
moons min_samples_leaf=5: train 0.952  test 0.944  leaves 16
rotate the features 45 degrees: test accuracy 0.944 -> 0.922 (drop +0.022)
regression tree depth 2:  4 distinct predictions, train MSE 0.0193
regression tree depth 3:  8 distinct predictions, train MSE 0.0091
regression tree depth 5: 32 distinct predictions, train MSE 0.0013

served: ['setosa', 'versicolor', 'virginica'] for [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 4.2, 1.5], [6.7, 3.0, 5.2, 2.3]]
```
