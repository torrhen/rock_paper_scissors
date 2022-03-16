import cv2 as cv
import numpy as np
import glob

def apply_edge_detection(image):
    # low_threshold = 255/3
    # high_threshold = 255
    image = cv.GaussianBlur(image, (3,3), 0)
    return cv.Sobel(src=image, ddepth=-1, dx=1, dy=1, ksize=5)
    # return cv.Canny(image, low_threshold, high_threshold)

def resize_and_normalize(image):
    new_width = 75
    new_height = 75
    image = cv.resize(image, (new_width, new_height))
    return image * 1/255.0

def get_image_file_paths_from(folder_path):
	file_paths = []
	for f in glob.glob(folder_path + "*.png"):
		file_paths.append(f)
	file_paths.sort()
	return file_paths

def read_image_data(file_paths):
	data = []
	for f in file_paths:
		image = cv.imread(f, cv.IMREAD_GRAYSCALE)
		# TODO: investigate function and optimal parameters
		image = apply_edge_detection(image)
		image = resize_and_normalize(image)
		data.append(image)
	# convert list to numpy array
	data = np.asarray(data)
	return data