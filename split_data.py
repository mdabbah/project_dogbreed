import pandas as pd
import numpy as np
import shutil, os


def move_examples(source_folder: str, target_folder: str,
                  new_order: np.ndarray, start_row: int,
                  row_count: int, label_table: pd.DataFrame):
    """
    moves .jpg images from rource folder to target folder, using the new random order
    of the given label table and saves a new label nable of the selected rows
    :param source_folder: where to take the images from
    :param target_folder: where to copy the images
    :param new_order: random premutation of label table rows
    :param start_row: start idx of the random premutaion to take rows from
    :param row_count: how many elements to take after the start row
    :param label_table: the label table
    :return: None
    """

    start_row = int(start_row)
    row_count = int(row_count)
    rows = new_order[start_row:start_row + row_count]
    new_lable_table = pd.DataFrame(label_table.iloc[rows])
    os.makedirs(target_folder, exist_ok=True)
    for row in new_lable_table.iterrows():
        filename_source = os.path.join(source_folder, row[1]['id'] + '.jpg')
        filename_target = os.path.join(target_folder, row[1]['id'] + '.jpg')
        shutil.copy(filename_source, filename_target)

    new_lable_table.to_csv(os.path.join(os.path.dirname(target_folder), os.path.basename(target_folder) + '_lables.csv'))


def split_data(dataset_folder: str, labels_file: str,
               training_percentage: float, validation_percentage: float):
    """
    splits the dataset int newly created folders according to the given percentages
    and saves the label tables for each folder
    :param dataset_folder:
    :param training_percentage:
    :param validation_percentage:
    :return:
    """

    np.random.seed(0)
    assert (training_percentage + validation_percentage < 1) and (training_percentage > 0)\
           and (validation_percentage > 0), " illegal data splitting "

    test_percentage = 1 - training_percentage - validation_percentage
    new_training_folder = os.path.join('./data', 'train_' + str(training_percentage))
    new_validation_folder = os.path.join('./data', 'valid_' + str(validation_percentage))
    new_test_folder = os.path.join('./data',  'test_' + str(test_percentage))

    label_table = pd.read_csv(labels_file)

    new_order = np.random.permutation(label_table.shape[0])
    num_training_examples = training_percentage * label_table.shape[0]
    num_validation_examples = validation_percentage * label_table.shape[0]
    num_test_examples = label_table.shape[0] - num_training_examples - num_validation_examples

    move_examples(dataset_folder, new_training_folder, new_order, 0, num_training_examples, label_table)
    move_examples(dataset_folder, new_validation_folder, new_order, num_training_examples, num_validation_examples, label_table)
    move_examples(dataset_folder, new_test_folder, new_order, num_validation_examples, num_test_examples, label_table)



if '__main__' == __name__:

    split_data(r'../train', '../labels_updated.csv', 0.8, 0.1)
