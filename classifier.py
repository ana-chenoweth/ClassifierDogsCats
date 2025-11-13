import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs
import os

# ============================================================
# 1. Dataset
# ============================================================

IMG_SIZE = (100, 100)

print("Cargando dataset desde TFDS...")
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

train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# ============================================================
# 2. Modelo Denso Mejorado
# ============================================================

model = keras.Sequential([
    layers.Flatten(input_shape=(100, 100, 1)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 3. Entrenar más tiempo
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25   # <-- Súbelo a 25, 40 o incluso 50
)

# ============================================================
# 4. Guardar modelo
# ============================================================

model.save("modelo_correcto.h5")
output_dir = "modelo_correcto_tfjs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tfjs.converters.save_keras_model(model, output_dir)
print("Modelo exportado a TensorFlow.js en:", output_dir)
