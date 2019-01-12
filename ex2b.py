import numpy as np
from pylab import imshow 
import matplotlib.pyplot as plt
import cv2


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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# paint image - function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paint_img(src_img,start_y, end_y, start_x, end_x ):
	for i in range(start_y, end_y):
		for j in range(start_x, end_x):
			src_img[i][j] = src_img[i][j]+1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#create rect border
rec_w = 200
rec_h = 200

start_point = 50
start_point_in = 50
rec_in_w =100
rec_in_h =100

rec = np.zeros((rec_w, rec_h))
points = np.zeros((rec_w, rec_h))
for x in range(start_point, start_point+rec_in_h):
    for y in range(start_point, start_point+rec_in_w):
        rec[x][y] = 255

for x in range(start_point+1, start_point+rec_in_h-1):
    for y in range(start_point+1, start_point+rec_in_w-1):
        rec[x][y] = 0

accumulator, thetas, rhos = hough_line(rec)

counter = 0
max_val = 0
for i  in range(len(accumulator)):
	for j in range(len(accumulator[0])):
		if(accumulator[i][j] >= max_val):
			max_val = accumulator[i][j]

max_counter = 0
array_rhos = []
array_thetas = []

for i  in range(len(accumulator)):
	for j in range(len(accumulator[0])):
		if(accumulator[i][j] == max_val):
			max_counter+= 1
			array_rhos.append(rhos[i])
			array_thetas.append(np.rad2deg(thetas[j]))
			# print("rho={0:.2f}, theta={1:.0f}".format(rhos[i],np.rad2deg(thetas[j])))

for i in range(len(array_thetas)):
	if(array_thetas[i] == 0):
		paint_img(points, start_y =0, end_y = rec_h, start_x = int(round(array_rhos[i])), end_x = int(round(array_rhos[i]))+1)
	if(array_thetas[i] == 90):
		paint_img(points, start_y = int(round(array_rhos[i])), end_y = int(round(array_rhos[i]))+1, start_x = 0, end_x = rec_w)
	if(array_thetas[i] == -90):
		paint_img(points, start_y = int(round(abs(array_rhos[i]))), end_y = int(round(abs(array_rhos[i])))+1, start_x = 0, end_x = rec_w)
		
max_pointes = np.max(points)
points_ = (points == max_pointes) * 255
points_ = points_.astype(np.uint8)

mask = np.array([[1, 1, 1],
[1, 1, 1],
[1, 1, 1]])

points_ = cv2.dilate(points_, mask,iterations = 1)

rec = rec.astype(np.uint8)
pointsColor = cv2.cvtColor(points_,cv2.COLOR_GRAY2RGB)
recColor = cv2.cvtColor(rec,cv2.COLOR_GRAY2RGB)


b,g,r = cv2.split(pointsColor)
g = np.zeros((rec_h, rec_w))
b = np.zeros((rec_h, rec_w))
redImg = np.dstack((r,g,b))
redImg = (redImg).astype(np.uint8)

src_with_red_cols = recColor
for x in range(rec_h):
    for y in range(rec_w):
        if(redImg[x][y][0]):
            src_with_red_cols[x][y] = redImg[x][y]

plt.figure("rec with red points :)")            
plt.imshow(src_with_red_cols)
plt.show()
