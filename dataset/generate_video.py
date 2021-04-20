
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf
import cv2
import argparse
from scipy import stats
import matplotlib.pyplot as plt

from src.visualize import vis_utils as vis
from src.io.psee_loader import PSEELoader

##############################################
'''
# https://github.com/bcrafton/event-based-vision/blob/power/power/agg/generate_events.py
def event2frame(events):
    x = events['x']
    y = events['y']
    p = events['p']
    
    tot = np.zeros((240, 304))
    for (j, i, k) in zip(y, x, p):
        tot[j, i] += 1
    tot = np.reshape(tot, -1)

    frame = np.zeros(shape=(240 * 304))
    for i, val in enumerate(tot):
        frame[i] = stats.percentileofscore(tot, val)
    frame = np.reshape(frame, (240, 304))

    frame = frame - np.min(frame)
    frame = frame / np.max(frame)
    frame = 1. - frame
    return frame
'''
##############################################
#'''
def event2frame(events):
    x = events['x']
    y = events['y']
    p = events['p'] * 2 - 1
    
    frame = np.zeros(shape=(240, 304))
    # frame[y, x] += np.ones_like(p)
    for (j, i, k) in zip(y, x, p):
        frame[j, i] += k

    frame = frame - np.min(frame) + 1
    frame = np.log10(frame)

    frame = frame / np.max(frame)
    frame = 1. - frame
    return frame
#'''
##############################################

def draw_box(image, box, cat):
    [x, y, w, h] = box
    pt1 = (int(x), int(y))
    pt2 = (int(x+w), int(y+h))
    cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=(1, 0, 0), thickness=2)

##############################################

def play_files_parallel(path, td_files, labels=None, delta_t=50000, skip=0):

    id = 0
    for video_idx in range(len(td_files)):
        print (video_idx)
    
        video = PSEELoader(td_files[video_idx])
        box_video = PSEELoader(td_files[video_idx].replace('_td.dat', '_bbox.npy'))
        
        frames = []
        while not video.done:
            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)
            frame = event2frame(events)
            frame = np.stack((frame, frame, frame), axis=-1)
            frames.append(frame)

            if len(boxes) or len(frames) == 16:
                for frame in frames:
                    for box in boxes:
                        box = np.array(list(box))
                        draw_box(frame, box[1:5], box[5])

                    plt.imsave('./frames/%d.png' % (id), frame)
                    id += 1

                frames = []

###########################################################

def collect_filenames(path):
    filenames = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == '.dat':
                filenames.append(os.path.join(subdir, file))

    filenames = sorted(filenames)
    return filenames

###########################################################

train_path = './src_data/'
# train_path = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/train/'

records = []
records = records + collect_filenames(train_path)

for record in records:
    print (record)

play_files_parallel('./train', records, skip=0, delta_t=50000)

###########################################################











    

