from tempfile import TemporaryFile
import numpy as np
from pylab import imshow, figure 
import matplotlib.pyplot as plt
from copy import copy
import cv2
from canny import *
from helpers import * 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def and_imgs(src, src2, dst):	
	for i  in range(len(src)):
		for j in range(len(src[0])):
			if(src[i][j]==src2[i][j]) and (src2[i][j] ==255):
				dst[i][j] = 255
	return dst
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

src_img = cv2.imread('sudoku-original.jpg', 0)
h = len(src_img)
w = len(src_img[0])

#threshold
threshold_img = (src_img > 48) * 255
threshold_img = ~threshold_img
threshold_img = threshold_img.astype(np.uint8)

#dilation
mask = np.array([[1, 1, 1 ,1, 1],
[1, 1, 1 ,1, 1],
[1, 1, 1 ,1, 1],
[1, 1, 1 ,1, 1],
[1, 1, 1, 1, 1]])

mask1 = np.array([[1, 1, 1],
[1, 1, 1],
[1, 1, 1]])

dilate_img_3 = cv2.dilate(threshold_img, mask1,iterations = 1)
dilate_img = cv2.dilate(dilate_img_3, mask,iterations = 1)
plt.figure()
plt.imshow(dilate_img, cmap = 'gray')


lines_img = np.zeros((h, w))
end_img   = np.zeros((h, w))
end_img2   = np.zeros((h, w))

#hough transform
accumulator, thetas, rhos = hough_line(dilate_img)
accumulator_array =np.array(accumulator).flatten()
sort = accumulator_array.sort()

for line in range(100):
	max_val = np.max(accumulator)
	for i  in range(len(accumulator)):
		for j in range(len(accumulator[0])):
			if(accumulator[i][j] == max_val):
				curr_rhos = rhos[i]
				curr_theta = thetas[j]
				cos_t = np.cos(curr_theta)
				sin_t = np.sin(curr_theta)
				accumulator[i][j] = 0

				for ii  in range(len(src_img)):
					for jj in range(len(src_img[0])):
						x = jj*cos_t+ii*sin_t
						if((abs(curr_rhos-x) <= 0.2) and (dilate_img_3[ii][jj])):
						    lines_img[ii][jj]	= 255
				
plt.figure("lines_img - before threshold")            
plt.imshow(lines_img, cmap = 'gray')


end_img = and_imgs(lines_img, dilate_img_3, end_img)

plt.figure("end_img - before threshold")            
plt.imshow(end_img, cmap = 'gray')

end_img = cv2.dilate(end_img, mask1,iterations = 1)
end_img2 = and_imgs(end_img, dilate_img_3, end_img2)

plt.figure("end_img2")            
plt.imshow(end_img2, cmap = 'gray')
plt.show()


