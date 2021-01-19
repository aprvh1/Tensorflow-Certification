import os #with_python
import zipfile #with_python
import random #with_python
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from shutil import copyfile #with_python

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

#mkdir -> another function to create dir..
#rm -rf '/tmp/PetImages/'
try:
    os.makedirs('/tmp/cats-v-dogs/training/cats/')
    os.makedirs('/tmp/cats-v-dogs/training/dogs/')
    os.makedirs('/tmp/cats-v-dogs/testing/cats/')
    os.makedirs('/tmp/cats-v-dogs/testing/dogs/')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    name_list = os.listdir(SOURCE)
    name_list = random.sample(name_list, len(name_list)) #shuffling list
    sz = int(len(name_list)*SPLIT_SIZE)+1
    list_train = name_list[0:sz]
    list_test = name_list[sz:-1]

    for item in list_train:
        if(os.path.getsize(SOURCE+item)>0):
            copyfile(SOURCE+item, TRAINING+item)

    for item in list_test:
        if(os.path.getsize(SOURCE+item)>0):
            copyfile(SOURCE+item, TESTING+item)

CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

TRAINING_DIR = '/tmp/cats-v-dogs/training/'
train_datagen = ImageDataGenerator(rescale=1/255.0)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)

VALIDATION_DIR = '/tmp/cats-v-dogs/testing/'
validation_datagen = ImageDataGenerator(rescale=1/255.0)
validation_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)

history = model.fit(train_generator,
                              epochs=15,
                              validation_data=validation_generator)


shutil.rmtree('/tmp/cats-v-dogs/')
