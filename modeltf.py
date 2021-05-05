import numpy as np
import utils
import matplotlib.pyplot as plt
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D



def create_model():
    model = Sequential()
    model.add(Conv2D(24, (5,5), kernel_initializer = 'he_uniform', 
                                input_shape = (64, 64, 1), 
                                activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 10, activation = 'softmax'))

    return model


def learn_model(model, plts = True):
    df = utils.load_dataset('test.csv')
    x = df.X
    X = np.array([utils.make_it_better(el) for el in x])
    Y = df.Y

    
    #X_train = tf.expand_dims(X, axis=-1)

    X_train = X.astype('float32')
    X_train /= 255
    X_train = X.reshape((-1, 64, 64, 1))
    Y_train = np_utils.to_categorical(Y, 10)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(Y_train.shape)
    trained_model_noreg = model.fit(X_train, Y_train, epochs=200, batch_size=100, validation_split=0.2)

    if plts:
        model_history = trained_model_noreg
        plt.plot(model_history.history['accuracy'])
        plt.plot(model_history.history['val_accuracy'])
        plt.title('Dokładność modelu')
        plt.ylabel('dokładność')
        plt.xlabel('epoki')
        plt.legend(['treningowe', 'walidacyjne'], loc='upper left')
        plt.show()
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('Funkcja błędu modelu')
        plt.ylabel('funkcja błędu')
        plt.xlabel('epoki')
        plt.legend(['treningowe', 'walidacyjne'], loc='upper left')
        plt.show()
    model.save('my_model')

    return model