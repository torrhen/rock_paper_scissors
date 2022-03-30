import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from images import read_image_data
from images import get_image_file_paths_from
from images import apply_edge_detection
from images import resize_and_normalize
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
# train cnn
results = cnn.fit(train_a_X, train_a_Y, epochs=20, validation_data=(test_a_X, test_a_Y))
# plot the classification accuracy of cnn for each epoch
plt.plot(results.history['accuracy'], label='Training')
plt.plot(results.history['val_accuracy'], label = 'Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# predict the target labels for unseen images and the confidence
predictions = cnn.predict(val_X)
for i in predictions:
	score = i
	print("This image is likely to be {} [{:.2f}% confidence]".format(all_labels[np.argmax(score)], 100 * np.max(score)))

# 2. open default camera

def calculate_view_frame_position(video_frame_width, video_frame_height):
    VIEW_FRAME_WIDTH = 300
    VIEW_FRAME_HEIGHT = 300
    VIEW_FRAME_THICKNESS = 2
    # calculate the top-left coordinates of the view frame
    x1 = ((video_frame_width - VIEW_FRAME_WIDTH) / 2) - VIEW_FRAME_THICKNESS
    y1 = ((video_frame_height - VIEW_FRAME_HEIGHT) / 2) - VIEW_FRAME_THICKNESS
    x1y1 = (int(x1), int(y1))
    # calculate the bottom-right coordinates of the view frame
    x2 = video_frame_width - x1
    y2 = video_frame_height - y1
    x2y2 = (int(x2), int(y2))

    return x1y1, x2y2


def draw_view_frame(video_frame):
    # dimensions of the frame to draw
    VIEW_FRAME_WIDTH = 300
    VIEW_FRAME_HEIGHT = 300
    VIEW_FRAME_THICKNESS = 2
    VIEW_FRAME_COLOUR = (0, 255, 0) # green

    video_frame_width = video_frame.shape[1]
    video_frame_height = video_frame.shape[0]

    x1y1, x2y2 = calculate_view_frame_position(video_frame_width, video_frame_height)

    # draw view frame on top of the video frame
    video_frame = cv.rectangle(video_frame, x1y1, x2y2, VIEW_FRAME_COLOUR, VIEW_FRAME_THICKNESS)
    return video_frame

isCameraOpen = False
# dimensions of each video frame
VIDEO_FRAME_WIDTH = 800
VIDEO_FRAME_HEIGHT = 600

# open default camera using the default API
camera = cv.VideoCapture(0)
if not camera.isOpened():
    # exit program if camera cannot be opened
    sys.exit("Failed to open camera.")

isCameraOpen = True
while isCameraOpen:
    # wait for a new frame from camera and store it into 'frame'
    ret, frame = camera.read()
    # check if frame was read
    if frame.all() == None:
        sys.exit("Failed to read video frame")
    
    # resize video frame
    frame = cv.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))
    # draw a centred, 300 x 300 view frame onto each video frame
    frame_edit = draw_view_frame(frame)
    # display video
    cv.namedWindow("Live", cv.WINDOW_AUTOSIZE)
    cv.imshow("Live", frame_edit)
    # close camera if ESC key is pressed
    keypressed = cv.waitKey(5)
    if keypressed == 27:
        camera.release()
        isCameraOpen = False
    if keypressed == 32:
        print("key pressed.")
        x1y1, x2y2 = calculate_view_frame_position(VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH)
        region = frame[x1y1[0]+2:x2y2[0]-2, x1y1[1]+2:x2y2[1]-2]
        # convert video frame region from colour to greyscale
        region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
        isCameraOpen = False

        # 3. process user image input
        x = apply_edge_detection(region)
        x = resize_and_normalize(x)
        cv.imshow("Live", x)
        cv.waitKey(0)
        # predict the label of the user input
        x = tf.convert_to_tensor([x], 1)
        pred = cnn.predict(x)
        print(pred)
        user_move = all_labels[np.argmax(pred)]
        print(user_move)

# 4. predict the label of the user input using cnn
# default player move
player_move = 'rock'

# 5. select computers random move
# default computer move
computer_move = 'scissors'

# 6. play game and decide winner
# print each player move
print("-------------------------")
print("Player move:\t" + player_move)
print("-------------------------")
print()
print("-------------------------")
print("Computer move:\t" + computer_move)
print("-------------------------")
print()
print("-------------------------")
# print the outcome of the game
if computer_move == "rock":
    if player_move == "rock":
        print("The game is a draw!")
    elif player_move == "scissors":
        print("The computer wins!")
    elif player_move == "paper":
        print("The player wins!")
elif computer_move == "paper":
    if player_move == "paper":
        print("The game is a draw!")
    elif player_move == "rock":
        print("The computer wins!")
    elif player_move == "scissors":
        print("The player wins!")
elif computer_move == "scissors":
    if player_move == "scissors":
        print("The game is a draw!")
    elif player_move == "paper":
        print("The computer wins!")
    elif player_move == "rock":
        print("The player wins!")
print("-------------------------")