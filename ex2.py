from tempfile import TemporaryFile
import numpy as np
from pylab import imshow, figure 
import matplotlib.pyplot as plt
from copy import copy
import cv2
from canny import *

# from scipy import imageio

img_name ='sudoku-original.jpg'
im = cv2.imread(img_name, 0)
img = im.astype('int32')
threshold_img = (img > 45) * 255
sigma = 0.2
t = 60
T = 100

img1 = gs_filter(threshold_img, sigma)
img2, D = gradient_intensity(img1)
img3 = suppression(img2, D)
img4, weak = threshold(img3, t, T)
img5 = tracking(img4, weak)

plt.imshow(~threshold_img, cmap = 'gray')
# plt.imshow(img5, cmap = 'gray')
plt.show()