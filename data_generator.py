from skimage.io import imread
from skimage.transform import resize
import numpy as np
import keras.utils
import os
import pandas as pd
from keras.applications.resnet50 import preprocess_input
import keras.applications as ka


class MYGenerator(keras.utils.Sequence):

    def __init__(self, train_folder, labels_file, batch_size, preprocess_fun_name):

        file_names = os.listdir(train_folder)
        image_file_names = [os.path.join(train_folder, im_file) for im_file in file_names]
        file_names = [os.path.splitext(os.path.basename(file))[0] for file in file_names]
        labels = pd.read_csv(labels_file)
        labels = [labels.loc[labels['id'] == file_names[idx]]['breed_id'] for idx in range(len(file_names))]
        labels = [int(round(l)) for l in labels]
        labels = np.array(labels, np.int)
        assert len(labels) == len(file_names), "not all images have labels"

        self.image_file_names, self.labels = image_file_names, labels
        self.batch_size = batch_size
        self.preprocess_fun = preprocess_fun_name
        print("created a generator from {:}".format(train_folder))

    def __len__(self):
        return np.int(np.ceil(len(self.image_file_names) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_file_names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        labels = keras.utils.to_categorical(batch_y, num_classes=120)

        base_models = {
            'resnet50': ka.resnet50.preprocess_input,

            'inception': ka.inception_v3.preprocess_input,

            'vgg19': ka.vgg19.preprocess_input
        }

        preprocess_fun = base_models[self.preprocess_fun]
        return np.array([
            resize(preprocess_fun(imread(file_name)), (224, 224))
            for file_name in batch_x]), labels
