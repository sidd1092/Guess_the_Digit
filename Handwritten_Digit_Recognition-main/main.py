#Installing libraries


#Setting tensorflow GPU
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

#Loading the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[2])

print(x_train.shape, y_train.shape)

#Preprocessing the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

#Creating the Model
model = Sequential()
model.add(Conv2D(32, kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Trainig the model
hist = model.fit(x_train, y_train, 
                 batch_size = 128, 
                 epochs = 10,
                 verbose = 1,
                 validation_data = (x_test, y_test))
                 
model.save('mnist.h5')
print("Saving the model as mnist.h5")

#Evaluating the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

