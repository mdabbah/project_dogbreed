from skimage.io import imread
from skimage.transform import resize
import numpy as np
import keras.utils
import os
import pandas as pd
from keras.applications.resnet50 import preprocess_input


class MYGenerator(keras.utils.Sequence):

    def __init__(self, train_folder, labels_file, batch_size):

        filenames = os.listdir(train_folder)
        image_filenames = [os.path.join(train_folder, im_file) for im_file in filenames]
        filenames = [os.path.splitext(file)[0] for file in filenames]
        labels = pd.read_csv(labels_file)
        labels = labels.loc[labels['id']==filenames]
        labels = np.array(labels['breed_id'], np.int)
        self.num_of_classes = len(set(labels))
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y_hot1mat = keras.utils.to_categorical(batch_y, 120)

        return np.array([
            resize(preprocess_input(imread(file_name)), (224, 224))
            for file_name in batch_x]), batch_y_hot1mat
