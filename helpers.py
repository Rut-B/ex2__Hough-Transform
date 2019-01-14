#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hough_line - function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# paint image - function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_img(src_img,start_y, end_y, start_x, end_x ):
	for i in range(start_y, end_y):
		for j in range(start_x, end_x):
			src_img[i][j] = src_img[i][j]+1
