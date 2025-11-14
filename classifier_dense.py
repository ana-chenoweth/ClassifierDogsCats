#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs

# ==========================
# CONFIGURACIÓN
# ==========================
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 30

print("📦 Cargando dataset 'cats_vs_dogs'...")
train_ds, val_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True
)

# ==========================
# PREPROCESAMIENTO + AUGMENTACIÓN
# ==========================
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # RGB normalizado [0,1]
    return image, label

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
], name="augmentation")

train_ds = (
    train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ==========================
# MODELO DENSO (REGULARNET)
# ==========================
inputs = keras.Input(shape=(100, 100, 3), name="input")
x = layers.Flatten()(inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs, name="DogCatDense")

# ==========================
# COMPILACIÓN + CALLBACKS
# ==========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

# ==========================
# ENTRENAMIENTO
# ==========================
print("🚀 Entrenando modelo Denso mejorado...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ==========================
# EXPORTACIÓN
# ==========================
print("💾 Exportando modelo Denso a TensorFlow.js...")
os.makedirs("web_model_dense", exist_ok=True)
tfjs.converters.save_keras_model(model, "web_model_dense")
print("✅ Exportación completada: carpeta 'web_model_dense/' lista para index.html")
