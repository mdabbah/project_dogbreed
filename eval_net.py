from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import sys,os
import numpy as np
import pandas as pd
from keras.applications.resnet50 import preprocess_input
from data_generator import  MYGenerator
from tensorflow.python.client import device_lib

if __name__ == '__main__':

    model = load_model(r'.\models\partially_trained_models\inception_resnet_v2\aaa.hdf5')

    print(device_lib.list_local_devices())

    if '-f' in sys.argv:
        image_path = sys.argv[sys.argv.index('-f') + 1]
        img_name = os.path.basename(image_path)
        img_name = os.path.splitext(img_name)[0]
        label_table = pd.read_csv(r'..\label_updated.csv')
        label = int(label_table.loc[label_table['id'] == img_name, 'breed_id'])

        print('our label is: ' + str(label))
        img = np.array([
            resize(preprocess_input(imread(file_name)), (224, 224))
            for file_name in [image_path]])
        predictions = model.predict(img)

        print('predicted top one: {0} confidence: {1}'.format(np.argmax(predictions), np.max(predictions)))
        print('argsort: ' + str(np.argsort(predictions)))

    if '-full_eval' in sys.argv:
        print('starting')
        batch_size = 32
        test_folder = r'./data/test_0.1'
        labels_file = './data/test_0.1_labels.csv'
        my_eval_batch_generator = MYGenerator(train_folder=test_folder, labels_file=labels_file,
                                              batch_size=batch_size, preprocess_fun_name='inception_resnet_v2')

        num_training_samples = os.listdir(test_folder).__len__()
        loss, acc = model.evaluate_generator(generator=my_eval_batch_generator,
                                             verbose=1,
                                             steps=1,
                                             use_multiprocessing=True,
                                             workers=4,
                                             max_queue_size=32)

        print('loss is:{0} acc is:{1}'.format(loss, acc))
