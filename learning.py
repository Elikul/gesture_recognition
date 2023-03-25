import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

DATA_PATH = "Hands_Data"
MODEL_PATH = "model"
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
label_map = {label: num for num, label in enumerate(numbers)}


def load_data():
    X = []
    y = []

    for number in numbers:
        data_path = os.path.join(DATA_PATH, number)
        json_datas = [f for f in os.listdir(data_path)]
        for data_file in json_datas:
            with open(os.path.join(DATA_PATH, number, data_file), 'r') as jf:
                data = json.load(jf)
                X.append(data['landmarks'])
                y.append(label_map[number])

    return X, y


def prepare_data():
    X, y = load_data()
    X = np.array(X)
    y = to_categorical(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    print(X_train.shape)
    return X_train, X_test, y_train, y_test


def create_model():
    model = Sequential()
    model.add(Dense(43 * 3, activation='relu', input_shape=(43, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def learn(model, X, y):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    history_fit = model.fit(X, y, epochs=100, batch_size=10, validation_split=0.2, callbacks=[reduce_lr, early_stop])
    return history_fit


def evaluate(model, test_x, test_y):
    scores = model.evaluate(test_x, test_y)
    print("Доля верных ответов на тестовых данных в процентах:", round(scores[1] * 100, 4))


def predict(model, test_x, test_y):
    predictions = model.predict(test_x)
    print(np.argmax(predictions[16]), np.argmax(test_y[16]))


def save_model(model):
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    model.save(os.path.join(MODEL_PATH, 'my_model'))
    model.save_weights(os.path.join(MODEL_PATH, 'weights'))


def draw_history():
    fig1 = plt.figure(figsize=(20, 10))
    plt.title("Train - Validation Accuracy")
    plt.plot(history_fit.history['accuracy'], label='train')
    plt.plot(history_fit.history['val_accuracy'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.savefig('accuracy.png')

    fig2 = plt.figure(figsize=(20, 10))
    plt.title("Train - Validation Loss")
    plt.plot(history_fit.history['loss'], label='train')
    plt.plot(history_fit.history['val_loss'], label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.savefig('loss.png')


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    model = create_model()
    history_fit = learn(model, X_train, y_train)

    draw_history()
    save_model(model)
    evaluate(model, X_test, y_test)
    predict(model, X_test, y_test)
