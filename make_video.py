import cv2
import numpy as np
import glob
from cv2 import VideoWriter, VideoWriter_fourcc

filenames = sorted(glob.glob('./results/*.jpg'))

img_array = []
for filename in filenames:
    img = cv2.imread(filename)
    img_array.append(img)

width = 2 * 288
height = 240
FPS = 15
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./video.avi', fourcc, float(FPS), (width, height))
 
for i in range(len(img_array)):
    video.write(img_array[i])

video.release()
