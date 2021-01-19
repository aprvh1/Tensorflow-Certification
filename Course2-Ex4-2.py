import os
import zipfile
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

local_zip = '/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

rock_dir='/tmp/rps/rock/'
paper_dir='/tmp/rps/paper/'
scissors_dir='/tmp/rps/scissors'

print("Rock: ",len(os.listdir(rock_dir)))
print("Paper: ",len(os.listdir(paper_dir)))
print("Scissors: ",len(os.listdir(scissors_dir)))

TRAINING_DIR='/tmp/rps/'
training_datagen = ImageDataGenerator(
    rescale=1/255.0,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=True,
    rotation_range=40,
    fill_mode='nearest'
)

training_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=128,                 #BATCH SIZE EFFECT EARLIER IT WAS 64 (ACC=0.7484), 126 is RECOMENDED
    class_mode='categorical',       #Important
    target_size=(150,150)
)

VALIDATION_DIR='/tmp/rps-test-set/'
validation_datagen = ImageDataGenerator(rescale=1/255.0)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical',       #Important
    batch_size=64
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),       #Important
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(training_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)