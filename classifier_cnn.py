#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs

IMG_SIZE = (100, 100)
BATCH_SIZE = 32

print("📦 Cargando dataset 'cats_vs_dogs'...")
train_ds, val_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True
)

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="augment")

train_ds = (
    train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# === Modelo CNN ===
inputs = keras.Input(shape=(100, 100, 1), name="input")
x = layers.Conv2D(32, (3,3), activation='relu', padding='same', name="conv2d")(inputs)
x = layers.MaxPooling2D(2,2, name="pool1")(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same', name="conv2d_1")(x)
x = layers.MaxPooling2D(2,2, name="pool2")(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same', name="conv2d_2")(x)
x = layers.MaxPooling2D(2,2, name="pool3")(x)
x = layers.Flatten(name="flatten")(x)
x = layers.Dense(128, activation='relu', name="dense")(x)
x = layers.Dropout(0.5, name="dropout")(x)
outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
model = keras.Model(inputs, outputs, name="DogCatCNN")

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])

callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

print("🚀 Entrenando modelo CNN...")
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

print("💾 Exportando a TensorFlow.js...")
os.makedirs("web_model_cnn", exist_ok=True)
tfjs.converters.save_keras_model(model, "web_model_cnn")
print("✅ Exportación completada: carpeta web_model_cnn/")
