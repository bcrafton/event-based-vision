import cv2
import numpy as np
import glob
from cv2 import VideoWriter, VideoWriter_fourcc

width = 304
height = 240
FPS = 25
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./video.avi', fourcc, float(FPS), (width, height))

# for i in range(10750, 15000):
for i in range(10750, 11750):
    filename = './dataset/frames/%d.png' % (i)
    frame = cv2.imread(filename)
    video.write(frame)

video.release()
