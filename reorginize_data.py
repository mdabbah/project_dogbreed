import pandas as pd
import numpy as np
import shutil, os


def copy_class(class_name: str, source_folder: str, target_folder: str,  labels_df: pd.DataFrame)->None:
    """
    copies images of members in given class from source folder to a newly made subdir of target_folder with
    the same name of the given class
    :param target_folder: where to save the newly made sub dir with all the copied images
    :param class_name: which class to copy
    :param source_folder: folder containing the images we want to organize (ex. ./data/train_0.8)
    :param labels_df: labels of images in the source folder (ex. ./data/train_0.8_labels.csv)
    :return: None
    """
    target_folder = os.path.join(target_folder, class_name)
    os.makedirs(target_folder, exist_ok=True)
    class_members = labels_df.loc[labels_df['breed'] == class_name, ['id']]

    for row_idx, row in class_members.iterrows():

        file_name_s = os.path.join(source_folder, row['id'])
        file_name_t = os.path.join(target_folder, row['id'])

        shutil.copy(file_name_s + '.jpg', file_name_t + '.jpg')

    print(f'done with {target_folder}')


if __name__ == '__main__':

    classes_df = pd.read_csv('./data/classes.csv')

    train_folder = './data/train_0.8'
    train_labels_df = pd.read_csv(train_folder + '_labels.csv')

    valid_folder = './data/valid_0.1'
    valid_labels_df = pd.read_csv(valid_folder + '_labels.csv')
    for row_idx, row in classes_df.iterrows():
        copy_class(row['class'], train_folder, './data_reorganized/train', train_labels_df)
        copy_class(row['class'], valid_folder, './data_reorganized/valid', valid_labels_df)
