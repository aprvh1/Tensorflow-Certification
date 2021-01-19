import tensorflow as tf
import numpy as np
from tensorflow import keras

x=[]
y=[]
for i in range(1,11):
    x.append(i)
    y.append(0.5+0.5*i)

print(x)
print(y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])
model.compile(loss='mean_squared_error',optimizer='sgd')
model.fit(x, y, epochs=10)
print(model.predict([7.0]))