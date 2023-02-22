# https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4
import keras.layers
import tensorflow
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import layers, models
from keras.callbacks import EarlyStopping
from PIL import Image
import os
import math

INPUT_DIRECTORY = "images/out"

inputs_a = [] # (n, 96, 96, 3)
inputs_b = [] # (n, 96, 96, 3)
labels = []  # (n, ...)

n = 0
for subdir in [f.path for f in os.scandir(INPUT_DIRECTORY) if f.is_dir()]:
    fragment_paths = os.listdir(subdir)
    amount_fragments_side = math.sqrt(len(fragment_paths))
    center_fragment_index = str(int(amount_fragments_side / 2))
    center_fragment_name = subdir.split('\\')[-1] + "_" + center_fragment_index + "_" + center_fragment_index + ".jpg"
    center_fragment = np.array(Image.open(os.path.join(subdir, center_fragment_name)))
    
    for file in os.listdir(subdir):
        if not file.endswith(center_fragment_name):
            ss = file.split('.')[0].split('_')
            fragment_index = int(ss[-2]) * amount_fragments_side + int(ss[-1])
            fragment_index = fragment_index if fragment_index <= int(math.pow(amount_fragments_side, 2) / 2) else fragment_index - 1
            lateral_fragment = np.array(Image.open(os.path.join(subdir, file)))

            inputs_a.append(center_fragment)
            inputs_b.append(lateral_fragment)
            labels.append(fragment_index)
    
    n += 1
    if n % 100 == 0: print(n)

    if (n >= 2000):
        break

print(1)

# Normalize
inputs_a = np.array(inputs_a) / 255.0
inputs_b = np.array(inputs_b) / 255.0
labels = to_categorical(labels, num_classes=int(math.pow(amount_fragments_side, 2) - 1)) # TODO: Should be 9 when allowing 'out' class

print(2)
# Shuffle
randomize = np.arange(len(inputs_a))
np.random.shuffle(randomize)
inputs_a = inputs_a[randomize]
inputs_b = inputs_b[randomize]
labels = labels[randomize]

print(3)
# Split train/test
train_test_frac = 0.8
split_index = int(len(inputs_a) * train_test_frac)
train_a = inputs_a[:split_index]
train_b = inputs_b[:split_index]
train_labels = labels[:split_index]
test_a = inputs_a[split_index:]
test_b = inputs_b[split_index:]
test_labels = labels[split_index:]

print(f"train_a, train_b, train_labels: {train_a.shape}, {train_b.shape}, {train_labels.shape}")
print(f"test_a, test_b, test_labels: {test_a.shape}, {test_b.shape}, {test_labels.shape}")

print(4)

# ===============================

# Network
single_image_shape = (96, 96, 3)

fen1 = VGG16(weights="imagenet", include_top=False, input_shape=single_image_shape)
fen1.trainable = False
fen1._name = "vgg16_1"
fen2 = VGG16(weights="imagenet", include_top=False, input_shape=single_image_shape)
fen2.trainable = False
fen2._name = "vgg16_2"

inputA = layers.Input(shape=single_image_shape)
inputB = layers.Input(shape=single_image_shape)

x = fen1(inputA)
x = keras.Model(inputs=inputA, outputs=x)
y = fen2(inputB)
y = keras.Model(inputs=inputB, outputs=y)
combined = layers.Multiply()([x.output, y.output])

flatten = layers.Flatten()(combined)
fc1 = layers.Dense(512, activation="relu")(flatten)
bn1 = layers.BatchNormalization()(fc1)
fc2 = layers.Dense(512, activation="relu")(bn1)
bn2 = layers.BatchNormalization()(fc2)
output = layers.Dense(8, activation="softmax")(bn2)

model = keras.Model(inputs=[x.input, y.input], outputs=output)
print(model.summary())
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

es = EarlyStopping(
    monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
)

# =====================

model.fit(
    [train_a, train_b],
    train_labels,
    epochs=50,
    validation_split=0.2,
    batch_size=32,
    callbacks=[es],
)

model.evaluate([test_a, test_b], test_labels)