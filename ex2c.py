from tempfile import TemporaryFile
import numpy as np
from pylab import imshow, figure 
import matplotlib.pyplot as plt
from copy import copy
import cv2
from canny import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hough_line - function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def hough_line(img):
	# Rho and Theta ranges
	thetas = np.deg2rad(np.arange(-90.0, 90.0))
	width, height = img.shape
	diag_len = int(np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
	rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
	# Cache some resuable values
	cos_t = np.cos(thetas)
	sin_t = np.sin(thetas)
	num_thetas = len(thetas)
	# Hough accumulator array of theta vs rho
	accumulator = np.zeros((2 * diag_len, num_thetas))
	y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges
	# Vote in the hough accumulator
	for i in range(len(x_idxs)):
		x = x_idxs[i]
		y = y_idxs[i]
		for t_idx in range(num_thetas):
			# Calculate rho. diag_len is added for a positive index
			rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
			accumulator[rho, t_idx] += 1
	return accumulator, thetas, rhos


src_img = cv2.imread('res.png', 0)
# src_img = cv2.imread('sudoku-original.jpg', 0)

#threshold

threshold_img = (src_img > 48) * 255
threshold_img = ~threshold_img
threshold_img = threshold_img.astype(np.uint8)

#dilation
mask = np.array([[1, 1, 1],
[1, 1, 1],
[1, 1, 1]])

dilate = cv2.dilate(threshold_img, mask,iterations = 1)

accumulator, thetas, rhos = hough_line(dilate)
plt.figure()
plt.imshow(accumulator, cmap = 'gray')
plt.figure()
plt.imshow(dilate, cmap = 'gray')
plt.show()

