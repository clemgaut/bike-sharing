# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:37:10 2014

@author: clemgaut
"""
import math
import matplotlib.pyplot as plt
import numpy as np

import utils

variables_to_plot = range(0, 9)

head = ["hour", "season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "casual",
        "registered", "count"]

y_index = 11

dim_plot = math.ceil(math.sqrt(len(variables_to_plot)))

train_content = utils.get_data('train.csv')

train_content = np.matrix(train_content)

for index, variable_to_plot in enumerate(variables_to_plot):
    value_table = {}

    column = np.ravel(train_content[:, variable_to_plot])

    for key in column:
        # increase count for the given variable value
        if key not in value_table.keys():
            value_table[key] = y_index
        else:
            value_table[key] += y_index

    plt.subplot(dim_plot, dim_plot, index+1)       
    plt.plot(value_table.keys(), value_table.values())
    plt.ylabel('Number of bike rentings')
    plt.xlabel(head[variable_to_plot])
    
plt.tight_layout()
plt.show()
