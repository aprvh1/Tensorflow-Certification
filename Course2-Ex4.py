import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(filename):
  # You will need to write code that will read the file passed
  # into this function. The first line contains the column headers
  # so you should ignore it
  # Each successive line contians 785 comma separated values between 0 and 255
  # The first value is the label
  # The rest are the pixel values for that picture
  # The function will return 2 np.array types. One with all the labels
  # One with all the images
  #
  # Tips:
  # If you read a full line (as 'row') then row[0] has the label
  # and row[1:785] has the 784 pixel values
  # Take a look at np.array_split to turn the 784 pixels into 28x28
  # You are reading in strings, but need the values to be floats
  # Check out np.array().astype for a conversion
    with open(filename) as training_file:
      images=[]
      labels=[]
      reader = csv.reader(training_file)
      next(reader, None)
      for row in reader:
        image= np.array(row[1:785]).astype(float)   #REALLY IMP HERE
        image= image.reshape(28,28)                 #REALLY IMP HERE
        images.append(image)
        label=int(row[0])
        labels.append(label)
    images=np.array(images)
    labels=np.array(labels)
    return images, labels

training_images, training_labels = get_data('/tmp/sign_lang_dataset-archive/sign_mnist_train.csv')
testing_images, testing_labels = get_data('/tmp/sign_lang_dataset-archive/sign_mnist_test.csv')

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

training_images = np.expand_dims(training_images, axis=-1)
testing_images = np.expand_dims(testing_images, axis=-1)

print(training_images.shape)
print(testing_images.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256,activation='relu'),
  tf.keras.layers.Dense(26,activation='softmax')
])
# Compile Model.
# WRONG :: model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model
history = model.fit(x=training_images,y=training_labels, epochs=20)

model.evaluate(testing_images, testing_labels)