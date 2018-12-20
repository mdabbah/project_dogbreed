from collections import OrderedDict

import pandas as pd
import numpy as np


def add_breed_id(filename: str = '..\labels.csv'):
    """
    adds breed_id column and saves the new label table under ../label_updated,csv
    :param filename: label table file with columns id ( file names) and breed ( string  breed)
    :return: None
    """
    # reads the label file and ads breed_id column
    labels = pd.read_csv(filename)
    labels['breed_id'] = np.nan
    classes = list(set(labels['breed']))

    for c in classes:
        labels.loc[labels['breed'] == c, ['breed_id']] = np.int(classes.index(c))

    labels.to_csv('..\label_updated.csv', index=False)


def generate_classes_file(filename: str = '..\labels_updated.csv'):
    """
    saves a new csv under .\data\classes.csv
    with columns class , id
    :param filename: label table file with columns 'breed and 'breed_id'
    :return: None
    """

    labels = pd.read_csv(filename)

    classes = {}

    for idx, row in labels.iterrows():
        _class = row['breed']
        _id = int(row['breed_id'])
        if _class not in classes.keys():
            classes[_class] = (_class, _id)
        elif classes[_class][1] != _id:
            raise ValueError(" same class different id")

    classes = OrderedDict(sorted(classes.items(), key=lambda t: t[1][1]))
    classes_df = pd.DataFrame(columns=['class', 'id'])
    for _class, pair in classes.items():
        classes_df = classes_df.append(pd.DataFrame([pair], columns=['class', 'id']), ignore_index=True)
        assert pair[0] == _class

    classes_df.to_csv('./data/classes.csv', index=False)


generate_classes_file()