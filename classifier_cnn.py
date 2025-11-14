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
EPOCHS = 25

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
    image = tf.cast(image, tf.float32) / 255.0  # RGB normalizado
    return image, label

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
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
# MODELO CNN PROFUNDO
# ==========================
inputs = keras.Input(shape=(100, 100, 3), name="input")

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs, name="DogCatCNN")

# ==========================
# COMPILACIÓN + CALLBACKS
# ==========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
]

# ==========================
# ENTRENAMIENTO
# ==========================
print("🚀 Entrenando modelo CNN mejorado...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ==========================
# EXPORTACIÓN
# ==========================
print("💾 Exportando modelo CNN a TensorFlow.js...")
os.makedirs("web_model_cnn", exist_ok=True)
tfjs.converters.save_keras_model(model, "web_model_cnn")
print("✅ Exportación completada: carpeta 'web_model_cnn/' lista para index.html")
