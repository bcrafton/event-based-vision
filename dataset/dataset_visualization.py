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
    # cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    
    # while all videos have something to read
    # while not sum([video.done for video in videos]):
    for idx in range(len(videos)):
    
        print (idx)
    
        xs = []
        frames = []
        dets = []

        while not videos[idx].done:

            # load events and boxes from all files
            events = videos[idx].load_delta_t(delta_t)
            box_events = box_videos[idx].load_delta_t(delta_t)
            evs, boxes = events, box_events

            # call the visualization functions
            frame = vis.make_binary_histo(evs, img=frame, width=width, height=height)

            assert (np.shape(frame) == (240, 304, 3))
            frame_preprocess = np.copy(frame[:, :, 0])
            # cv2 takes things as {W,H} even when array is sized {H,W}
            frame_preprocess = cv2.resize(frame_preprocess, (288, 240))
            frames.append(frame_preprocess)
            # NEED TO RESIZE THE DETECTIONS!!!
            # CAN WE JUST DO 288*304 TO THAT LAST DIMENSION.
            if len(boxes):
                # assert (np.all(frame[:, :, 0] == frame[:, :, 1]))
                frames = frames[-12:]
                frames = np.stack(frames, axis=-1)
                for box in boxes:
                    box[1] = round(box[1] * (288 / 304))
                    box[3] = round(box[3] * (288 / 304))
                if np.shape(frames) == (240, 288, 12):
                    xs.append(frames)
                    dets.append(np.copy(boxes))
                frames = []

            vis.draw_bboxes(frame_preprocess, boxes)

            # display the result
            '''
            if len(boxes):
                cv2.imshow('out', frame_preprocess)
                cv2.waitKey(1)
            '''

        print (np.shape(np.array(xs)))
        dataset = {'x': np.array(xs), 'y': dets}
        np.save('./data/%d' % (idx), dataset)

'''
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='visualize one or several event files along with their boxes')
    parser.add_argument('records', nargs="+", help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('-s', '--skip', default=0, type=int, help="skip the first n microseconds")
    parser.add_argument('-d', '--delta_t', default=20000, type=int, help="load files by delta_t in microseconds")
    return parser.parse_args()
'''

if __name__ == '__main__':

    records = []
    for subdir, dirs, files in os.walk('./src_data'):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == '.dat':
                records.append(os.path.join(subdir, file))

    print (records)
    play_files_parallel(records, skip=0, delta_t=20000)
    
    
    
    
    
    
    
    
    
