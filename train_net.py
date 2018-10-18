import keras.applications as ka
from keras.layers import Dense
from keras.optimizers import SGD

num_classes = 120
weights = r'C:\Users\mdabb\Desktop\ProjectB\project dog breeds\keras-pretrained-models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
net = ka.ResNet50(include_top=False, weights=weights, classes=num_classes, pooling='avg')
net.add(Dense(num_classes, activation='softmax'))

# Learning rate is changed to 0.001
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
net.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])