import cv2
import numpy as np
import glob
from cv2 import VideoWriter, VideoWriter_fourcc

# TODO: MAKE SURE THIS IS THE WRITE SIZE.
width = 288*2+5
height = 240
FPS = 25
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./video.avi', fourcc, float(FPS), (width, height))

# for i in range(5000, 10000):
# anytime after 9 seconds really.
# 25*37 : 25*67
# 25*96 : 25*124

for i in range(5925, 6675):
    filename = '../results_train/%d.png' % (i)
    frame = cv2.imread(filename)
    video.write(frame)

for i in range(7400, 8150):
    filename = '../results_train/%d.png' % (i)
    frame = cv2.imread(filename)
    video.write(frame)

video.release()
