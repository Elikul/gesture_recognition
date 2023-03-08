import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

tf.compat.v1.disable_eager_execution()

TRAIN_PATH = "resources/train"
TEST_PATH = "resources/test"
BATCH_SIZE = 10
LABEL_TYPE = "categorical"
DIMENSION_IMAGE = (64, 64)
SHUFFLE = True

RELU_ACTIVATION = "relu"
SOFTMAX_ACTIVATION = "softmax"
SAME_PADDING = "same"
VALID_PADDING = "valid"
KERNEL_SIZE = (3, 3)
STRIDES = 2
POOL_SIZE = (2, 2)
INPUT_SHAPE = (64, 64, 3)

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=TRAIN_PATH,
                                                                                             target_size=DIMENSION_IMAGE,
                                                                                             class_mode=LABEL_TYPE,
                                                                                             batch_size=BATCH_SIZE,
                                                                                             shuffle=SHUFFLE)
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=TEST_PATH,
                                                                                             target_size=DIMENSION_IMAGE,
                                                                                             class_mode=LABEL_TYPE,
                                                                                             batch_size=BATCH_SIZE,
                                                                                             shuffle=SHUFFLE)

word_dict = {0: 'One', 1: 'Two', 2: 'Three', 3: 'Four', 4: 'Five', 5: 'Six', 6: 'Seven', 7: 'Eight',
             8: 'Nine'}


# построение изображений
def plotting_images(images_arr):
    fig, axes = plt.subplots(1, 9, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# imgs, labels = next(train_batches)
# plotting_images(imgs)
# print(imgs.shape)
# print(labels)


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation=RELU_ACTIVATION, input_shape=INPUT_SHAPE))
    model.add(MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES))
    model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation=RELU_ACTIVATION, padding=SAME_PADDING))
    model.add(MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES))
    model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation=RELU_ACTIVATION, padding=VALID_PADDING))
    model.add(MaxPool2D(pool_size=POOL_SIZE, strides=STRIDES))
    model.add(Flatten())
    model.add(Dense(64, activation=RELU_ACTIVATION))
    model.add(Dense(128, activation=RELU_ACTIVATION))
    # model.add(Dropout(0.2))
    model.add(Dense(128, activation=RELU_ACTIVATION))
    # model.add(Dropout(0.3))
    model.add(Dense(9, activation=SOFTMAX_ACTIVATION))
    return model


def train_model():
    model = create_model()
    model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=test_batches)
    print("History ", history2.history)
    imgs, labels = next(test_batches)
    scores = model.evaluate(imgs, labels, verbose=0)
    print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    model.save('best_model.h5')

    predictions = model.predict(imgs, verbose=0)
    for ind, i in enumerate(predictions):
        print(word_dict[np.argmax(i)], end='   ')

    plotting_images(imgs)
    for i in labels:
        print(word_dict[np.argmax(i)], end='   ')
