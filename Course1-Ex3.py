import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

mnist = mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_test = x_test/255.0
x_train = x_train/255.0

print(len(x_train))

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')>0.99):
            print("\nAccuracy reached 99%")
            self.model.stop_training = True

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

model = tf.keras.models.Sequential([
    keras.layers.Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128,activation='relu'),
    keras.layers.Dense(units=10,activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbk=myCallback()
model.fit(x_train,y_train,epochs=20,callbacks=[callbk])
model.evaluate(x_test,y_test)