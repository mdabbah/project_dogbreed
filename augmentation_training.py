from glob import glob
import keras.applications as ka
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from data_generator import MYGenerator
import os
import numpy as np, pandas as pd
import smtplib
import keras.callbacks
from keras.layers import Input
import time, datetime
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


base_models = {
    'resnet50': (ka.ResNet50,  r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.resnet50.preprocess_input),

    'inception': (ka.InceptionV3, r'..\keras-pretrained-models\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.inception_v3.preprocess_input),

    'vgg19': (ka.VGG19,  r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.vgg19.preprocess_input),

    'inception_resnet_v2': (ka.InceptionResNetV2, ' ',
                 ka.inception_resnet_v2.preprocess_input),

    'xception': (ka.Xception, r'..\keras-pretrained-models\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.xception.preprocess_input),

    'mobilenet_v2': (ka.MobileNetV2,  r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.mobilenet_v2.preprocess_input)
}


class Logger:

    def __init__(self, file_name):
        self.log_file_name = file_name

    def log(self, msg):

        with open(self.log_file_name, 'a') as f:
            time_str = str(datetime.datetime.today())
            f.write('\n\n' + time_str + '\n'+ str(msg))


def save_stats(msg):

    with open('stats.txt', 'a') as f:
        f.write(str(msg))


if __name__ == '__main__':

    logger = Logger('log.txt')

    num_classes = 120
    image_input_size = (224, 224)
    train_epochs = 150
    batch_size = 32
    seed = 42
    classes = pd.read_csv('./data/classes.csv')
    classes = classes['class'].tolist()

    for base_mode_name in base_models.keys():

        if base_mode_name != 'inception_resnet_v2':
            continue

        base_model_constructor = base_models[base_mode_name][0]
        weights_path = 'imagenet'  # base_models[base_mode_name][1]
        base_model_pre_process = base_models[base_mode_name][2]

        input_tensor = Input(shape=(*image_input_size, 3))  # this assumes K.image_data_format() == 'channels_last'
        base_model = base_model_constructor(input_tensor=input_tensor,
                                            include_top=False, weights=weights_path,
                                            classes=num_classes, pooling='avg')
        x = base_model.output
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False

        # Learning rate is changed to 0.001
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        train_folder = r'./data_reorganized/train'
        train_labels_file = './data/train_0.8_labels.csv'
        my_training_batch_generator = ImageDataGenerator(shear_range=0.2,
                                                         brightness_range=[0.5, 1.5],
                                                         fill_mode='nearest',
                                                         zoom_range=[0.9, 1.25],
                                                         rotation_range=90,
                                                         horizontal_flip=True,
                                                         preprocessing_function=base_model_pre_process)\
                                                        .flow_from_directory(
                                                            directory=train_folder,
                                                            target_size=image_input_size,
                                                            color_mode="rgb",
                                                            batch_size=batch_size,
                                                            class_mode="categorical",
                                                            shuffle=True,
                                                            seed=seed, classes=classes
                                                        )
        valid_folder = r'./data_reorganized/valid'
        valid_labels_file = './data/valid_0.1_labels.csv'
        my_validation_batch_generator = ImageDataGenerator(shear_range=0.2,
                                                           brightness_range=[0.5, 1.5],
                                                           fill_mode='nearest',
                                                           zoom_range=[0.9, 1.25],
                                                           rotation_range=45,
                                                           horizontal_flip=True,
                                                           preprocessing_function=base_model_pre_process)\
                                                        .flow_from_directory(
                                                            directory=valid_folder,
                                                            target_size=image_input_size,
                                                            color_mode="rgb",
                                                            batch_size=batch_size,
                                                            class_mode="categorical",
                                                            shuffle=True,
                                                            seed=seed, classes=classes
                                                        )

        test_folder = r'./data/test_0.1'
        test_labels_file = './data/test_0.1_labels.csv'
        my_test_batch_generator = MYGenerator(train_folder=test_folder, labels_file=test_labels_file,
                                              batch_size=batch_size, preprocess_fun_name=base_mode_name)

        partially_trained_folder = os.path.join(r'./models_aug/partially_trained_models', base_mode_name)
        os.makedirs(partially_trained_folder, exist_ok=True)
        save_models_names_format = os.path.join(partially_trained_folder, base_mode_name +
                                                '{epoch:02d} -loss {loss:.3f} -acc{acc:.4f} -'
                                                'val_loss {val_loss: .3f} -val_acc {val_acc:.4f}.hdf5')
        check_point_callback = keras.callbacks.ModelCheckpoint\
            (save_models_names_format)

        num_training_samples = glob(train_folder+'/*/*').__len__()
        num_valid_samples = glob(valid_folder+'/*/*').__len__()
        logger.log('starting training with base model {:}'.format(base_mode_name))
        model.fit_generator(generator=my_training_batch_generator,
                            validation_data=my_validation_batch_generator,
                            steps_per_epoch=(num_training_samples // batch_size),
                            epochs=train_epochs,
                            verbose=1,
                            use_multiprocessing=False,
                            workers=4,
                            max_queue_size=32, validation_steps=(num_valid_samples//batch_size),
                            callbacks=[check_point_callback])

        logger.log('finished training with {:}'.format(base_mode_name))
        # logger.log('weights for model{:} before saving:'.format(base_mode_name))
        # logger.log(model.weights)

        loss, acc = model.evaluate_generator(generator=my_test_batch_generator,
                                             verbose=1,
                                             steps=1,
                                             use_multiprocessing=True,
                                             workers=16,
                                             max_queue_size=32)

        logger.log('model {:} evaluated on test set: acc{:}, loss{:}'.format(base_mode_name, acc, loss))

        trained_folder = os.path.join(r'./models_aug/trained_models', base_mode_name)
        os.makedirs(trained_folder, exist_ok=True)
        os.makedirs(partially_trained_folder, exist_ok=True)
        model_save_path = os.path.join(trained_folder, 'trained_{:}_{:}.h5'.format(base_mode_name,
                                                                                   datetime.datetime.today())
                                       .replace(':', '-'))
        model.save(model_save_path)

        del model

        model = load_model(model_save_path)
        logger.log('saved model .. now deleting it .. loading and printing weights')
        # logger.log(model.weights)

        test_folder = r'./data/test_0.1'
        test_labels_file = './data/test_0.1_labels.csv'
        my_test_batch_generator = MYGenerator(train_folder=test_folder, labels_file=test_labels_file,
                                              batch_size=batch_size, preprocess_fun_name=base_mode_name)


        loss, acc = model.evaluate_generator(generator=my_test_batch_generator,
                                             verbose=1,
                                             steps=1,
                                             use_multiprocessing=True,
                                             workers=16,
                                             max_queue_size=32)

        logger.log('loaded model {:} evaluated on test set: acc{:}, loss{:}'.format(base_mode_name, acc, loss))

        try:

            # Gmail Sign In
            gmail_sender = 'project.doogbreed@gmail.com'
            gmail_passwd = 'Pass4D0GBreed'

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.login(gmail_sender, gmail_passwd)

            server.sendmail(gmail_sender, ['m.m.dabbah@gmail.com'], 'training has finished')
            print('email sent')
        except:
            print('error sending mail')