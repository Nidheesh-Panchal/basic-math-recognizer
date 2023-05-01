import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

base_dir = os.getcwd()
src_dir = os.path.join(base_dir, "src")
model_dir = os.path.join(src_dir, "model")
data_dir = os.path.join(src_dir, "data")

batch_size = 128
img_row = 28
img_col = 28
channel = 1
print(f"Input image dimensions: {img_row} x {img_col} x {channel}")

labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'add': 10, 'dec': 11, 'div': 12, 'eq': 13, 'mul': 14, 'sub': 15, 'x':16, 'y': 17, 'z': 18}
label = list(labels.keys())

num_classes = len(label)
# print("Labels dict: ", labels)
# print("Labels list: ", label)
print("Num of classes: ", num_classes)

df = pd.read_csv(os.path.join(data_dir, "data.csv"), index_col=0)

X = df.values[:,:-1]
Y = df.values[:,-1]
X = X.reshape(df.shape[0], img_row, img_col, channel).astype('float32')
X = X / 255
Y = tf.keras.utils.to_categorical(Y, num_classes)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15)

model = tf.keras.models.Sequential()

model.add(tf.keras.Input(shape=(img_row, img_col, channel)))

model.add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = (img_row, img_col, channel), padding = "same"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation = "softmax"))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

print("\nSearching for model file if already present")
if("model.h5" in os.listdir(model_dir)):
    print("Model file already present")
    exit()

print("Training model")

epochs = 100
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train, 
                    epochs=epochs,
                    callbacks = [early_stop],
                    validation_data = (X_val, Y_val),
                    verbose = 1)

print("Saving model")

model.save(os.path.join(model_dir, "model.h5"))
model.save_weights(os.path.join(model_dir, "model_weights.h5"))

print("Model saved")