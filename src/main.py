import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from images import read_image_data
from images import get_image_file_paths_from

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
train_r_Y = tf.ones([train_r_X.shape[0], 1]) * 0.0
train_p_Y = tf.ones([train_p_X.shape[0], 1]) * 1.0
train_s_Y = tf.ones([train_s_X.shape[0], 1]) * 2.0
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
test_r_Y = tf.ones([test_r_X.shape[0], 1]) * 0.0
test_p_Y = tf.ones([test_p_X.shape[0], 1]) * 1.0
test_s_Y = tf.ones([test_s_X.shape[0], 1]) * 2.0
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

# display the first random training image and corresponding target label
plt.imshow(train_a_X[0], cmap='gray')
print(train_a_Y[0])
plt.show()