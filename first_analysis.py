# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:37:10 2014

@author: clemgaut
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import metrics

import numpy as np

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

# build classifier
clf = RandomForestRegressor(n_estimators=10)

# cross-validation evaluation
scores = cross_validation.cross_val_score(clf, data, result_vect, cv=5, scoring=rmsle_scorer)
scores = -scores

print "RMSLE:" + str(np.mean(scores)) + " +- " + str(np.std(scores))

test_data = utils.get_data("test.csv")
clf.fit(data, result_vect)
pred_test = clf.predict(test_data)

utils.write_predictions(pred_test, "res_RF10.csv")
