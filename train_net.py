import keras.applications as ka
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

num_classes = 120
weights = r'..\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = ka.ResNet50(include_top=False, weights=weights, classes=num_classes, pooling='avg')
x = base_model.output
x = GlobalAveragePooling2D()(x)
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

model.fit_generator()