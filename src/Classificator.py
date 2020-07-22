import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
from PIL import Image

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

print('START')

BATCH_SIZE = 32
IMG_SIZE = 224
NCLASSES = 107

data_dir = 'dataset'

datagen = ImageDataGenerator(horizontal_flip=True,
                             validation_split=0.2)  # Todo configure rescale=1./255 + data augmentation
train_iterator = datagen.flow_from_directory('dataset/', class_mode='categorical', target_size=(IMG_SIZE, IMG_SIZE),
                                             batch_size=BATCH_SIZE, subset='training')  # Todo check the options
test_iterator = datagen.flow_from_directory('dataset/', class_mode='categorical', target_size=(IMG_SIZE, IMG_SIZE),
                                            batch_size=BATCH_SIZE, subset='validation')  # Todo check the options

model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                          # should try tf.keras.applications.vgg19.VGG19
                                          include_top=False,
                                          weights='imagenet')

batchX, batchy = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

feature_batch = model(batchX)
print(feature_batch.shape)

# Feature extraction

model.trainable = False

model.summary()  # show the architecture

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(NCLASSES, activation='softmax')  # really softmax?
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)


model = tf.keras.Sequential([
    model,
    global_average_layer,
    prediction_layer
])

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),  # Try ADAM
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

initial_epochs = 10
validation_steps = 20

# loss0, accuracy0 = model.fit_generator(validation_batches, steps=validation_steps)

# model.fit_generator()
loss0, accuracy0 = model.fit_generator(train_iterator, epochs=10)
