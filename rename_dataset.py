import os
import sys
import pandas as pd


dataset_base_dir = r'../dataset'
counter = 0
labels = pd.read_csv('../labels')

for f in os.listdir(dataset_base_dir):

    if os.path.isdir(f):
        new_folder = os.path.join(dataset_base_dir, f + '_renamed')
        old_folder = os.path.join(dataset_base_dir, f)
        os.mkdir(new_folder)
        num_files = copyfiles(new_folder, old_folder, counter, labels)
        counter += num_files