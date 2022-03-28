import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from images import read_image_data
from images import get_image_file_paths_from
from tensorflow.keras import layers, models

all_labels = ['rock', 'paper', 'scissors']

# training data
# read training image data from folder and store in numpy array
train_r_X = read_image_data(get_image_file_paths_from("../res/data/train/rock/"))
train_p_X = read_image_data(get_image_file_paths_from("../res/data/train/paper/"))
train_s_X = read_image_data(get_image_file_paths_from("../res/data/train/scissors/"))
# convert to tensors
train_r_X = tf.convert_to_tensor(train_r_X)
train_p_X = tf.convert_to_tensor(train_p_X)
train_s_X = tf.convert_to_tensor(train_s_X)
# generate training target labels
train_r_Y = tf.ones([train_r_X.shape[0], 1], dtype=tf.int32) * 0
train_p_Y = tf.ones([train_p_X.shape[0], 1], dtype=tf.int32) * 1
train_s_Y = tf.ones([train_s_X.shape[0], 1], dtype=tf.int32) * 2
# concat training images into a single tensor
train_a_X = tf.concat([train_r_X, train_p_X, train_s_X], 0)
# concat training target labels into a single tensor
train_a_Y = tf.concat([train_r_Y, train_p_Y, train_s_Y], 0)
# shuffle the order of training images and corresponding target labels
indices = tf.range(start=0, limit=tf.shape(train_a_X)[0], dtype=tf.int32)
indices_sh = tf.random.shuffle(indices)
# training images and labels have corresponding shuffled indices
train_a_X = tf.gather(train_a_X, indices_sh)
train_a_Y = tf.gather(train_a_Y, indices_sh)

# test data
# read test image data from folder and store in numpy array
test_r_X = read_image_data(get_image_file_paths_from("../res/data/test/rock/"))
test_p_X = read_image_data(get_image_file_paths_from("../res/data/test/paper/"))
test_s_X = read_image_data(get_image_file_paths_from("../res/data/test/scissors/"))
# convert to tensors
test_r_X = tf.convert_to_tensor(test_r_X)
test_p_X = tf.convert_to_tensor(test_p_X)
test_s_X = tf.convert_to_tensor(test_s_X)
# generate test target labels
test_r_Y = tf.ones([test_r_X.shape[0], 1], dtype=tf.int32) * 0
test_p_Y = tf.ones([test_p_X.shape[0], 1], dtype=tf.int32) * 1
test_s_Y = tf.ones([test_s_X.shape[0], 1], dtype=tf.int32) * 2
# concat test images into a single tensor
test_a_X = tf.concat([test_r_X, test_p_X, test_s_X], 0)
# concat test target labels into a single tensor
test_a_Y = tf.concat([test_r_Y, test_p_Y, test_s_Y], 0)
# shuffle the order of test images and corresponding target labels
indices = tf.range(start=0, limit=tf.shape(test_a_X)[0], dtype=tf.int32)
indices_sh = tf.random.shuffle(indices)
# test images and labels have corresponding shuffled indices
test_a_X = tf.gather(test_a_X, indices_sh)
test_a_Y = tf.gather(test_a_Y, indices_sh)

# # display the first random training image and corresponding target label
# plt.imshow(train_a_X[0], cmap='gray')
# print(all_labels[int(train_a_Y[0])])
# plt.show()

# validation data
val_X = read_image_data(get_image_file_paths_from("../res/data/validation/"))
# convert to tensor
val_X = tf.convert_to_tensor(val_X)

# train cnn
cnn = models.Sequential()
cnn.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(75, 75, 1)))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(84, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Flatten())
cnn.add(tf.keras.layers.Dropout(0.1))
cnn.add(layers.Dense(60, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.1))
cnn.add(layers.Dense(10, activation='relu'))
cnn.add(layers.Dense(3, activation='softmax'))
cnn.summary()

cnn.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
