"""
Utilities: helper functions for the ensemble example.
"""

import json
from sklearn.model_selection import GroupShuffleSplit

def import_json(f):
    '''
    (str) -> list of lists

    Given a a string leading to a valid JSON file will return a list
    of lists with the terminal nodes being dictionaries that can later
    be converted to data frames or matrices.
    '''
    json_data = open(f)
    data = json.load(json_data)
    json_data.close()
    return data

def import_cv_vars(f):
    '''
    (str) -> list of lists

    Given a a string leading to a valid JSON file will return a list
    of lists with the terminal nodes being dictionaries that can later
    be converted to data frames or matrices.
    '''
    return import_json(f)

def split_groups(df, n_splits, test_size, groups):
    """
    Creates a list of training sets and test sets of size n_splits
    where the tests are created based on knocking out groups defined
    by the groups variable.

    :param df: data frame to split
    :n_splits: number of times to split training-test
    :test_size: percent of data for testing
    :groups: column name in df to split by
    """
    ss = GroupShuffleSplit(n_splits, test_size)

    # create a shuffle split generator
    train_list = []
    test_list = []

    # subset data frames to a training set and a test set
    for train_index, test_index in ss.split(df, groups=df[groups]):
        train_list.append(df.iloc[list(train_index),:])
        test_list.append(df.iloc[list(test_index),:])

    return train_list, test_list

