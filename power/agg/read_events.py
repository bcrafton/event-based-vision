"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import tensorflow as tf
import cv2
import argparse

import matplotlib.pyplot as plt

from src.visualize import vis_utils as vis
from src.io.psee_loader import PSEELoader

##############################################

def play_files_parallel(path, td_files, labels=None, delta_t=50000, skip=0):

    # frame = np.zeros((240, 304, 3), dtype=np.uint8)

    counter = 0
    for filename in td_files:

        video = PSEELoader(filename)
        box_video = PSEELoader(filename.replace('_td.dat', '_bbox.npy'))

        xs = []
        ys = []
        ts = []
        ps = []
        while not video.done:

            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)

            # frame = vis.make_binary_histo(events, img=frame, width=304, height=240)
            # plt.imshow(frame)
            # plt.show()

            # print (events['x'])
            # print (events['y'])
            # print (events['ts'])
            # print (events['p'])

            # print (events.names)
            # print (events.fields)

            # print (events.dtype)
            # [('ts', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')]

            # print (dir(events))
            # assert (False)

            x = events['x']
            y = events['y']
            t = events['ts']
            p = events['p']
            # print (np.shape(x))

            if len(boxes):
                xs.append(x)
                ys.append(y)
                ts.append(t)
                ps.append(p)

        xs = xs[-12:]
        ys = ys[-12:]
        ts = ts[-12:]
        ps = ps[-12:]

        if len(xs) == 12:
            '''
            print (np.shape(xs))
            print (np.shape(ys))
            print (np.shape(ts))
            print (np.shape(ps))
            print ()
            '''

            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            ts = np.concatenate(ts)
            ps = np.concatenate(ps)

            data = {'x': xs, 'y': ys, 't': ts, 'p': ps}
            np.save('./data/' + str(counter), data)
            counter += 1

###########################################################

train_path = '../../dataset/train_src/'

###########################################################

records = []
for subdir, dirs, files in os.walk(train_path):
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.dat':
            records.append(os.path.join(subdir, file))

records = sorted(records)
for record in records:
    print (record)

play_files_parallel('./train', records, skip=0, delta_t=20000)

###########################################################
    
    
    
    
    
    
    
