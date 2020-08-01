import cv2
import numpy as np
import glob
from cv2 import VideoWriter, VideoWriter_fourcc

width = 2 * 288
height = 240
FPS = 15
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./video.avi', fourcc, float(FPS), (width, height))


'''
for i in range(384, 2500, 1):
    filename = './results/%d.jpg' % (i)
    frame = cv2.imread(filename)
    video.write(frame)
'''

for i in range(384, 534, 1):
    filename = './results/%d.jpg' % (i)
    frame = cv2.imread(filename)
    video.write(frame)
    
for i in range(1434, 1634, 1):
    filename = './results/%d.jpg' % (i)
    frame = cv2.imread(filename)
    video.write(frame)

video.release()
