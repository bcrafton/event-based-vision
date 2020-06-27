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
import cv2
import argparse

import matplotlib.pyplot as plt

from src.visualize import vis_utils as vis
from src.io.psee_loader import PSEELoader

def play_files_parallel(td_files, labels=None, delta_t=50000, skip=0):
    """
    play simultaneously files and their boxes in a rectangular format
    """
    # open the video object for the input files
    videos = [PSEELoader(td_file) for td_file in td_files]
    # use the naming pattern to find the corresponding box file
    box_videos = [PSEELoader(td_file.replace('_td.dat', '_bbox.npy')) for td_file in td_files]

    height, width = videos[0].get_size()

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for video_idx in range(len(videos)):
        print (video_idx)
        frame_idx = 0
        frames = []
        while not videos[video_idx].done:

            # load events and boxes from all files
            events = videos[video_idx].load_delta_t(delta_t)
            box_events = box_videos[video_idx].load_delta_t(delta_t)
            evs, boxes = events, box_events

            # call the visualization functions
            frame = vis.make_binary_histo(evs, img=frame, width=width, height=height)

            assert (np.shape(frame) == (240, 304, 3))
            frame_preprocess = np.copy(frame[:, :, 0])
            # cv2 takes things as {W,H} even when array is sized {H,W}
            frame_preprocess = cv2.resize(frame_preprocess, (288, 240))
            frames.append(frame_preprocess)
            if len(boxes):
                frames = frames[-12:]
                frames = np.stack(frames, axis=-1)
                for box in boxes:
                    box[1] = round(box[1] * (288 / 304))
                    box[3] = round(box[3] * (288 / 304))
                if np.shape(frames) == (240, 288, 12):
                    sample = {'x': frames, 'y': boxes}
                    np.save('./data/%d_%d' % (video_idx, frame_idx), sample)
                frames = []
                frame_idx += 1

if __name__ == '__main__':

    records = []
    for subdir, dirs, files in os.walk('./src_data'):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == '.dat':
                records.append(os.path.join(subdir, file))

    print (records)
    play_files_parallel(records, skip=0, delta_t=20000)
    
    
    
    
    
    
    
    
    
