import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import numpy as np

mnist = mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

#print(type(y_train))
#print(np.unique(y_train))

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy')>0.99):
            print("\nAccuracy reached 99%")
            self.model.stop_training = "True"

callback_ap=myCallback()

#print (x_train[0])
#print (y_train[0])
model = tf.keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=10 , activation= tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callback_ap])
model.evaluate(x_test, y_test)