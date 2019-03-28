import keras.applications as ka
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from glob import glob

"""
this module is to try different augmentation parameters and visualize their
 effect on images feed to the training process 
"""
if __name__ == '__main__':

    base_model_pre_process = ka.inception_resnet_v2.preprocess_input
    file_name = './example_img.jpg'
    image_input_size = (224, 224)
    seed = 42
    save_agu_dir = './data_augmented_visualize/example_1'
    os.makedirs(save_agu_dir, exist_ok=True)
    img = np.array([resize(base_model_pre_process(imread(file_name)), (224, 224))])
    generator = ImageDataGenerator(shear_range=0.2,
                                   brightness_range=[0.5, 1.5],
                                   fill_mode='nearest',
                                   zoom_range=[0.9, 1.25],
                                   rotation_range=90,
                                   horizontal_flip=True,
                                   preprocessing_function=base_model_pre_process).\
                                   flow(img, [0],
                                        save_to_dir=save_agu_dir,
                                        shuffle=True, batch_size=img.shape[0],
                                        seed=seed, save_prefix='rename_me_')

    for idx in range(30):
        next(generator)
        src = glob(save_agu_dir + '/rename_me_*')[0]
        os.rename(src, src.replace('rename_me', f'sample_{idx}_'))

