
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

##############################################

def draw_box(image, box, cat, color):
    [x, y, w, h] = box
    pt1 = (int(x), int(y))
    pt2 = (int(x+w), int(y+h))
    cv2.rectangle(image, pt1, pt2, (1, 0, 0), 2)
    # label = 'Human' if cat == 1 else 'Car'
    # cv2.putText(image, label, (int(x), int(y)), 0, 0.3, (1, 0, 0))

##############################################

def play_files_parallel(path, td_files, labels=None, delta_t=50000, skip=0):

    id = 0
    for video_idx in range(len(td_files)):
        print (video_idx)
    
        video = PSEELoader(td_files[video_idx])
        box_video = PSEELoader(td_files[video_idx].replace('_td.dat', '_bbox.npy'))
        
        events_list = []
        while not video.done:
            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)
            events_list.append(events)
            
            if len(boxes) and len(events_list) >= 12:
                id += 1
                if id < 8150: continue

                events_list = events_list[-12:]
                
                for F in range(12):
                    frame = event2frame(events_list[F])                
                    frame = np.stack((frame, frame, frame), axis=-1)
                    for box in boxes:
                        box = np.array(list(box))
                        box[1] = round(box[1] * (288 / 304))
                        box[3] = round(box[3] * (288 / 304))
                        draw_box(frame,  box[1:5], box[5], None)
                        plt.imsave('./frames/%d_%d.png'   % (id, F), frame)

                events_list = []
                # id += 1
                assert (id < 8160)

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

# train_path = './src_data/'
train_path = '/home/bcrafton3/Data_HDD/prophesee-automotive-dataset/train/'

records = []
records = records + collect_filenames(train_path)

for record in records:
    print (record)

play_files_parallel('./train', records, skip=0, delta_t=20000)

###########################################################











    

