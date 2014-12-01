# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:37:10 2014

@author: clemgaut
"""
import csv
import math
import matplotlib.pyplot as plt


def get_hour(date):
    """
    Get the hour from a date formatted as dd/mm/yyyy hh:mm:ss
    Return hh as an int
    """
    time = date.split(' ')[2]    
    hour = time.split(':')[0]
    
    return int(hour)

variables_to_plot = range(1,8)
dim_plot = math.ceil(math.sqrt(len(variables_to_plot)))


for index,variable_to_plot in enumerate(variables_to_plot):
    value_table = {}
    csv_file = open('train.csv', 'rb')
    train_content = csv.reader(csv_file)
    head = train_content.next()
    for row in train_content:
        #increase count for the given variable value
        key = int(float(row[variable_to_plot]))
        if key not in value_table.keys():
            value_table[key] = int(row[-1])
        else:
            value_table[key] += int(row[-1])

    plt.subplot(dim_plot, dim_plot, index+1)       
    plt.plot(value_table.keys(), value_table.values())
    plt.ylabel('Number of bike rentings')
    plt.xlabel(head[variable_to_plot])
    
    csv_file.close()
    
plt.tight_layout()
plt.show()
        
