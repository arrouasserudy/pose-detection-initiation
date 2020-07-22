import cv2
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from src.classes import yoga_classes

BATCH_SIZE = 32
IMG_SIZE = 224
NCLASSES = 107

data_dir = 'dataset/'
checkpoint_path = "weights/cp.ckpt"


class Classificator:
    def __init__(self, learning_rate=0.001, from_weights=False):
        self.data_generator = ImageDataGenerator(horizontal_flip=True, validation_split=0.2)
        mobile_net_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                       include_top=False,
                                       weights='imagenet')
        mobile_net_model.trainable = False
        self.model = tf.keras.Sequential([
            mobile_net_model,
            GlobalAveragePooling2D(),
            Dense(NCLASSES, activation='softmax')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),  # Try ADAM
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

        # Create a callback that saves the model's weights
        self.callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=True,
                                                           verbose=1)

        if from_weights:
            self.model.load_weights(checkpoint_path)


    def get_batch_iterator(self, subset=None):
        return self.data_generator.flow_from_directory(data_dir, class_mode='categorical',
                                                       target_size=(IMG_SIZE, IMG_SIZE),
                                                       batch_size=BATCH_SIZE,
                                                       subset=subset)

    def train(self, batch_iterator, epochs=1):
        loss0, accuracy0 = self.model.fit_generator(batch_iterator, epochs=epochs, callbacks=[self.callback]) #Todo use fit, deprecated

    def predict(self, frame):
        if type(frame) is str:
            frame = cv2.imread(frame)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = np.reshape(frame, [1, IMG_SIZE, IMG_SIZE, 3])
        prediction = self.model.predict(frame)
        classIndex = prediction.argmax(axis=-1)
        return yoga_classes[classIndex[0]]
