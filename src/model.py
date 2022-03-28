import cv2 as cv
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import random
from images import *



# class_names = ["rock", "paper", "scissors"]

# validation_image_data = read_image_data(get_image_file_paths_from("../res/data/validation/"))
# validation_data = Dataset()
# validation_data.data = tf.convert_to_tensor(validation_image_data)

# predictions = model.predict(validation_data.data)

# for i in predictions:
# 	score = i
# 	print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# 	)

# isCameraOpen = False

# def draw_view_frame(video_frame):
#     # dimensions of the frame to draw
#     VIEW_FRAME_WIDTH = 300
#     VIEW_FRAME_HEIGHT = 300
#     VIEW_FRAME_THICKNESS = 2
#     VIEW_FRAME_COLOUR = (0, 255, 0) # green

#     video_frame_width = video_frame.shape[0]
#     video_frame_height = video_frame.shape[1]

#     # calculate the top-left and bottom right coordinates of the view frame
#     x1y1 = (int((video_frame_width - VIEW_FRAME_WIDTH) / 2) - VIEW_FRAME_THICKNESS, int((video_frame_height - VIEW_FRAME_HEIGHT) / 2) - VIEW_FRAME_THICKNESS)
#     x2y2 = (int(video_frame_width - x1y1[0]), int(video_frame_height - x1y1[1]))

#     # draw view frame on top of the video frame
#     video_frame = cv.rectangle(video_frame, x1y1, x2y2, VIEW_FRAME_COLOUR, VIEW_FRAME_THICKNESS)
#     return video_frame


# def launch_camera():
#     # dimensions of each video frame
#     VIDEO_FRAME_WIDTH = 600
#     VIDEO_FRAME_HEIGHT = 600

#     # open default camera using the default API
#     camera = cv.VideoCapture(0)
#     if not camera.isOpened():
#         # exit program if camera cannot be opened
#         sys.exit("Failed to open camera.") 
#     isCameraOpen = True

#     while isCameraOpen:
#         # wait for a new frame from camera and store it into 'frame'
#         ret, frame = camera.read()
#         # check if frame was read
#         if frame.all() == None:
#             sys.exit("Failed to read video frame")
        
#         # resize video frame
#         frame = cv.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))    
#         # draw a centred, 300 x 300 view frame onto each video frame
#         frame = draw_view_frame(frame)
#         # display video
#         cv.namedWindow("Live", cv.WINDOW_AUTOSIZE)
#         cv.imshow("Live", frame)
#         # close camera if ESC key is pressed
#         if cv.waitKey(5) == 27:
#             camera.release()
#             isCameraOpen = False

# launch_camera()
# x = apply_edge_detection(x)
# x = resize_and_normalize(x)
# x = tf.convert_to_tensor(x)
        # conert video frame to greyscale
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


# print(x.shape)

        # keypressed = cv.waitKey(0)
        # if keypressed == 27:
        #     # cv.Mat player_image = video_frame(cv.Rect((video_frame.cols - border_width) / 2, (video_frame.rows - border_height) / 2, border_width, border_height));
        #     # cv.cvtColor(player_image, player_image, cv.COLOR_BGR2GRAY);
        #     # cv.hconcat(video_frame, player_image, video_frame);
        #     isCameraOpen = False
        # # ENTER key
        # elif keypressed == 32:
        #     region = region[start[0]+2:end[0]-2, start[1]+2:end[1]-2]
        #     # convert video frame region from colour to greyscale
        #     region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)

        #     # display frame region
        #     cv.namedWindow("Region", cv.WINDOW_AUTOSIZE)
        #     cv.imshow("Region", region)   
        #     cv.waitKey(0)
        #     isCameraOpen = False

        #     return region


# actions = ['Rock', 'Paper', 'Scissors']
# # predict the video frame using the CNN and store the result as the player move.
# move = model.predict(x)
# player_move = move[0]
# # select computer move with uniform randomness
# computer_move = random.choice(actions)

# print("#########################")
# print("Player move:\t" + player_move)
# print("#########################")
# print()
# print("#########################")
# print("Computer move:\t" + computer_move)
# print("#########################")
# print()
# print("#########################")
# # play the game and decide the winner
# if computer_move == "Rock":
#     if player_move == "Rock":
#         print("The game is a draw!")
#     elif player_move == "Scissors":
#         print("The computer wins!")
#     elif player_move == "Paper":
#         print("The player wins!")
# elif computer_move == "Paper":
#     if player_move == "Paper":
#         print("The game is a draw!")
#     elif player_move == "Rock":
#         print("The computer wins!")
#     elif player_move == "Scissors":
#         print("The player wins!")
# elif computer_move == "Scissors":
#     if player_move == "Scissors":
#         print("The game is a draw!")
#     elif player_move == "Paper":
#         print("The computer wins!")
#     elif player_move == "Rock":
#         print("The player wins!")
# print("#########################")