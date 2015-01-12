__author__ = 'clemg_000'

# This script builds random forest using both casual and registered counts

from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

import utils

registered_index = 10
casual_index = 9

# ignore results columns for training data
ignore_columns = [9, 10, 11]

data = utils.get_data('train.csv')

data = np.matrix(data)

registered_result = data[:, registered_index]
registered_result = np.ravel(registered_result)

casual_result = data[:, casual_index]
casual_result = np.ravel(casual_result)

# remove columns not needed for training
data = np.delete(data, ignore_columns, 1)

# define custom corer for RMSLE
rmsle_scorer = metrics.make_scorer(utils.get_RMSLE, greater_is_better=False)

n_trees_list = [10, 50, 100]
cv_means_reg = []
cv_stds_reg = []
cv_means_cas = []
cv_stds_cas = []

# build classifier
for n_trees in n_trees_list:
    clf = RandomForestRegressor(n_estimators=n_trees)

    # cross-validation evaluation for registered
    scores = cross_validation.cross_val_score(clf, data, registered_result, cv=10, scoring=rmsle_scorer)
    scores = -scores
    cv_means_reg.append(np.mean(scores))
    cv_stds_reg.append(np.std(scores))

    # cross-validation evaluation for casual
    scores = cross_validation.cross_val_score(clf, data, casual_result, cv=10, scoring=rmsle_scorer)
    scores = -scores
    cv_means_cas.append(np.mean(scores))
    cv_stds_cas.append(np.std(scores))

# Plots mean and std depending on number of trees
plt.subplot(221)
plt.plot(n_trees_list, cv_means_reg)
plt.xlabel("Number of trees")
plt.ylabel("Mean RMSLE for registered")

plt.subplot(222)
plt.plot(n_trees_list, cv_stds_reg)
plt.xlabel("Number of trees")
plt.ylabel("Standard dev RMSLE for registered")

plt.subplot(223)
plt.plot(n_trees_list, cv_means_cas)
plt.xlabel("Number of trees")
plt.ylabel("Mean RMSLE for casual")

plt.subplot(224)
plt.plot(n_trees_list, cv_stds_cas)
plt.xlabel("Number of trees")
plt.ylabel("Standard dev RMSLE for casual")

plt.tight_layout()
plt.show()

# Make predictions according to best result
cv_summary = np.add(cv_means_cas, cv_means_reg)

clf_reg = RandomForestRegressor(n_estimators=n_trees_list[np.argmin(cv_summary)])
clf_cas = RandomForestRegressor(n_estimators=n_trees_list[np.argmin(cv_summary)])

test_data = utils.get_data("test.csv")

clf_reg.fit(data, registered_result)
clf_cas.fit(data, casual_result)

pred_test_reg = clf_reg.predict(test_data)
pred_test_cas = clf_cas.predict(test_data)

pred_test = np.add(pred_test_reg, pred_test_cas)

utils.write_predictions(pred_test, "res_RF_reg_casual.csv")
