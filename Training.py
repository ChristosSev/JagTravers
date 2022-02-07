from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers

train_dir = '/Users/Christos/PycharmProjects/kerasnew/Christos/EXPERIMENT/TRAIN_K/'
validation_dir = '/Users/Christos/PycharmProjects/kerasnew/Christos/EXPERIMENT/VALID_K/'
image_size = 224

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all the layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax')) # h softmax ?????   #### na dokimasw kai gia 1 output

# Show a summary of the model. Check the number of trainable parameters
model.summary()


train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 10
val_batchsize = 1

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',shuffle=True)

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['acc'])

# Train the Model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=20 ,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=32,
      verbose=1)

# Save the Model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("modelglllyyyom.h5")   #model 10 gia 200 epochs
model.save_weights("LuISSSSSSito.h5")
print("Saved model to disk")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


