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

# Gmail Sign In
gmail_sender = 'project.doogbreed@gmail.com'
gmail_passwd = 'Pass4D0GBreed'

server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.login(gmail_sender, gmail_passwd)

if __name__ == '__main__':

    num_classes = 120
    weights = r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = ka.ResNet50(include_top=False, weights=weights, classes=num_classes, pooling='avg')
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
    train_folder = r'..\train'
    labels_file = '..\label_updated.csv'
    my_training_batch_generator = MYGenerator(train_folder=train_folder, labels_file=labels_file,
                                              batch_size=batch_size)

    check_point_callback = keras.callbacks.ModelCheckpoint\
        (r'..\trained _resnet_partial_2\resnet50.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5')
    num_training_samples = os.listdir(train_folder).__len__()

    model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch=(num_training_samples // batch_size),
                        epochs=100,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=16,
                        max_queue_size=32,
                        callbacks=[check_point_callback])

    model.save(r'..\trained_resnet.h5')

    try:
        server.sendmail(gmail_sender, ['m.m.dabbah@gmail.com'], 'training has finished')
        print('email sent')
    except:
        print('error sending mail')