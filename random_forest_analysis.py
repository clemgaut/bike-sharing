# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:37:10 2014

@author: clemgaut
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

import utils

# Index of column with results
result_index = 11
# ignore results columns for training data
ignore_columns = [9, 10, 11]

data = utils.get_data('train.csv')

data = np.matrix(data)

# get result vector
result_vect = data[:, result_index]
result_vect = np.ravel(result_vect)

# remove columns not needed for training
data = np.delete(data, ignore_columns, 1)

# define custom corer for RMSLE
rmsle_scorer = metrics.make_scorer(utils.get_RMSLE, greater_is_better=False)

n_trees_list = [10, 50, 100, 200]
cv_means = []
cv_stds = []

# build classifier
for n_trees in n_trees_list:
    clf = RandomForestRegressor(n_estimators=n_trees)

    # cross-validation evaluation
    scores = cross_validation.cross_val_score(clf, data, result_vect, cv=5, scoring=rmsle_scorer)
    scores = -scores
    cv_means.append(np.mean(scores))
    cv_stds.append(np.std(scores))


# Plots mean and std depending on number of trees
plt.subplot(211)
plt.plot(n_trees_list, cv_means)
plt.xlabel("Number of trees")
plt.ylabel("Mean RMSLE")

plt.subplot(212)
plt.plot(n_trees_list, cv_stds)
plt.xlabel("Number of trees")
plt.ylabel("Standard dev RMSLE")

plt.tight_layout()
plt.show()

# Make predictions according to best result
clf = RandomForestRegressor(n_estimators=n_trees_list[np.argmax(cv_means)])
test_data = utils.get_data("test.csv")
clf.fit(data, result_vect)
pred_test = clf.predict(test_data)

utils.write_predictions(pred_test, "res_RF10.csv")
