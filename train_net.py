import keras.applications as ka
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from data_generator import MYGenerator
import os
import smtplib
import keras.callbacks
from keras.layers import Input
import time
from keras.models import load_model


# Gmail Sign In
gmail_sender = 'project.doogbreed@gmail.com'
gmail_passwd = 'Pass4D0GBreed'

server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.login(gmail_sender, gmail_passwd)

nets = {'resnet50': (ka.ResNet50, ka.resnet50.preprocess_input, ka.resnet50.decode_predictions),
            'inception': (ka.InceptionV3, ka.inception_v3.preprocess_input, ka.inception_v3.decode_predictions),
            'vgg19': (ka.VGG19, ka.vgg19.preprocess_input, ka.vgg19.decode_predictions)}

base_models = {
    'resnet50': (ka.ResNet50,  r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.resnet50.preprocess_input),

    'inception': (ka.InceptionV3,  r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.inception_v3.preprocess_input),

    'vgg19': (ka.VGG19,  r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                 ka.vgg19.preprocess_input)
}


class Logger:

    def __init__(self, file_name):
        self.log_file_name = file_name

    def log(self, msg):

        with open(self.log_file_name, 'a') as f:
            f.write(str(msg))


def save_stats(msg):

    with open('stats.txt', 'a') as f:
        f.write(str(msg))

if __name__ == '__main__':

    logger = Logger('log.txt')


    num_classes = 120
    base_mode_name = 'resnet50'

    base_model_constructor = base_models[base_mode_name][0]
    weights_path =  base_models[base_mode_name][1]
    base_model_pre_process = base_models[base_mode_name][2]

    # batch_size = 32
    # train_folder = r'./data/train_0.8'
    # train_labels_file = './data/train_0.8_labels.csv'
    # my_training_batch_generator = MYGenerator(train_folder=train_folder, labels_file=train_labels_file,
    #                                           batch_size=batch_size, preprocess_fun=base_model_pre_process)

    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
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


    batch_size = 32
    train_folder = r'./data/train_0.8'
    train_labels_file = './data/train_0.8_labels.csv'
    my_training_batch_generator = MYGenerator(train_folder=train_folder, labels_file=train_labels_file,
                                              batch_size=batch_size, preprocess_fun_name=base_mode_name)
    valid_folder = r'./data/valid_0.1'
    valid_labels_file = './data/valid_0.1_labels.csv'
    my_validation_batch_generator = MYGenerator(train_folder=valid_folder, labels_file=valid_labels_file,
                                                batch_size=batch_size, preprocess_fun_name=base_mode_name)

    test_folder = r'./data/valid_0.1'
    test_labels_file = './data/valid_0.1_labels.csv'
    my_test_batch_generator = MYGenerator(train_folder=test_folder, labels_file=test_labels_file,
                                          batch_size=batch_size, preprocess_fun_name=base_mode_name)

    partially_trained_folder = os.path.join(r'./models/partially_trained_models', base_mode_name)
    os.makedirs(partially_trained_folder, exist_ok=True)
    save_models_names_format = os.path.join(partially_trained_folder, base_mode_name +
                                            '{epoch:02d}-{loss:.2f}-{acc:.2f}-{loss: .3f}.hdf5')
    check_point_callback = keras.callbacks.ModelCheckpoint\
        (save_models_names_format)

    num_training_samples = os.listdir(train_folder).__len__()
    logger.log('{:} starting training with base model {:}'.format(time.time(), base_mode_name))
    model.fit_generator(generator=my_training_batch_generator,
                        validation_data=my_validation_batch_generator,
                        steps_per_epoch=(num_training_samples // batch_size),
                        epochs=100,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=16,
                        max_queue_size=32,
                        callbacks=[check_point_callback])

    logger.log('{:} finished training with {:}'.format(time.time(), base_mode_name))
    logger.log('weights for model{:} before saving:'.format(base_mode_name))
    logger.log(model.weights)

    loss, acc = model.evaluate_generator(generator=my_test_batch_generator,
                                         verbose=1,
                                         steps=1,
                                         use_multiprocessing=True,
                                         workers=16,
                                         max_queue_size=32)

    logger.log('model {:} evaluated on test set: acc{:}, loss{:}'.format(base_mode_name, acc, loss))

    trained_folder = os.path.join(r'./models/trained_models', base_mode_name)
    os.makedirs(partially_trained_folder, exist_ok=True)
    model_save_path = os.path.join(trained_folder, 'trained_{:}_{:}.h5'.format(base_mode_name, time.time()))
    model.save(model_save_path)

    del model

    model = load_model(model_save_path)
    logger.log(' saved model .. now deleting it .. loding and printing weights')
    logger.log(model.weights)

    try:
        server.sendmail(gmail_sender, ['m.m.dabbah@gmail.com'], 'training has finished')
        print('email sent')
    except:
        print('error sending mail')