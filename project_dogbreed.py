import pandas as pd
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import keras
import keras.applications as ka
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from typing import List


dog_breeds = pd.read_csv('../dog_breeds.csv')


def load_net(argv: List[str]):
    """
    loads one of the predefined nets
    :param argv: list of given comand line arguments
    :return: net keras.engine.training.Model
    """

    nets = {'resnet50': (ka.ResNet50, ka.resnet50.preprocess_input, ka.resnet50.decode_predictions),
            'inception': (ka.InceptionV3, ka.inception_v3.preprocess_input, ka.inception_v3.decode_predictions),
            'vgg19': (ka.VGG19, ka.vgg19.preprocess_input, ka.vgg19.decode_predictions)}

    if '-net' in argv:
        net_name = argv[argv.index('-net') + 1]
        if net_name in nets.keys():
            print('loading net: ' + net_name)
            return nets[net_name]
        else:
            print('net unavailable, net options are' + nets.keys().__str__())
    elif '-load_net' in argv:
        net_name = argv[argv.index('-load_net') + 1]
        print('loading given net:' + os.path.basename(net_name))
        return keras.models.load_model(net_name), ka.resnet50.preprocess_input, ka.resnet50.decode_predictions
    else:
        print(' loading default option: resnet50')
        return ka.resnet50.ResNet50(weights='imagenet'), ka.resnet50.preprocess_input, ka.resnet50.decode_predictions


def load_image(argv: List[str]):
    """
    loades given image from flags or loads defualt image
    :return:
    """

    if '-img' in argv:
        img_path = argv[argv.index('-img') + 1]
        try:
            print('loading image: ' + os.path.basename(img_path))
            img = image.load_img(img_path, target_size=(224, 224))
            return image.img_to_array(img)
        except:
            print('could not load image')
    print('loading default image')
    img_path = r'..\dataset\train\0a077ea0c8fa54d95f75e690b2c8196b.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    return image.img_to_array(img)


if __name__ == '__main__':

    net, preprocess_input, decode_predictions = load_net(sys.argv)

    # loading and showing the image
    img = load_image(sys.argv)

    fig, ax = plt.subplots(1, figsize=(12, 10))
    plt.imshow(img / 255.)
    ax.imshow(img / 255.)
    ax.axis('off')
    plt.show()

    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    preds = net.predict(x)
    print(decode_predictions(preds, top=5))