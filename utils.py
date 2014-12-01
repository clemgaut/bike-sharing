# -*- coding: utf-8 -*-
__author__ = 'clemgaut'

import csv

import numpy as np


def get_hour(date):
    """
    Get the hour from a date formatted as dd/mm/yyyy hh:mm:ss
    Return hh as an int
    """
    time = date.split(' ')[1]
    hour = time.split(':')[0]

    return int(hour)


def get_data(file_name):
    """
    Get the data from the dataset and apply preprocessing to extract hours.
    """
    csv_file = open(file_name, 'rb')
    train_content = csv.reader(csv_file)

    # ignore header
    train_content.next()

    # preprocessing functions for each column index
    # Several preprocessing can be defined for each column.
    # A new variable is associated to EACH preprocessing function
    preproc_funcs = {0: ['get_hour']}

    # Read data from file, store it as an integer
    data = []
    for row in train_content:
        data_row = []
        for n, col in enumerate(row):
            # if the current column requires preprocessing functions, apply them
            if preproc_funcs.has_key(n):
                # Each preprocessing give a new column
                for preproc_func in preproc_funcs[n]:
                    func = globals().get(preproc_func)
                    data_row.append(int(float(func(col))))
            # If no preprocessing, do nothing
            else:
                data_row.append(int(float(col)))

        data.append(data_row)

    csv_file.close()

    return data


def get_datetimes(file_name):
    """ Get the datetimes from the excel file """
    csv_file = open(file_name, 'rb')
    file_content = csv.reader(csv_file)

    # ignore header
    file_content.next()

    datetimes = []

    for row in file_content:
        datetimes.append(row[0])

    csv_file.close()

    return datetimes


def write_predictions(pred, filename="pred.csv"):
    """
    Write the predictions in the filename according to Kaggle format.
    :param pred: The prediction vector
    :param filename: Filename of the output file
    """
    output_file = open(filename, "wb")
    writer = csv.writer(output_file)
    datetimes = get_datetimes("test.csv")

    writer.writerow(["datetime", "count"])

    for index, count in enumerate(pred):
        writer.writerow([datetimes[index], int(count)])

    output_file.close()


def get_RMSLE(pred, truth):
    """
    Return RMSLE from the prediction and the expected answer.
    """
    assert len(pred) == len(truth)
    diff_vect = np.log(pred + 1) - np.log(truth + 1)
    diff_sum = np.sum(np.power(diff_vect, 2))
    return np.sqrt(diff_sum / len(pred))