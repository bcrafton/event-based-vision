

import numpy as np
import tensorflow as tf
from dataset.src.metrics.coco_eval import _coco_eval
from draw_boxes import draw_box
import matplotlib.pyplot as plt

####################################

def soft_nms(boxes, confs):
    N = len(boxes)
    tag = [False] * N
    for i in range(N):
        for j in range(i + 1, N):
            ###
            box1 = boxes[i]
            box2 = boxes[j]
            ###
            yA = max(box1[0],           box2[0])
            yB = min(box1[0] + box1[2], box2[0] + box2[2])
            ###
            xA = max(box1[1],           box2[1])
            xB = min(box1[1] + box1[3], box2[1] + box2[3])
            inter = max(0, yB - yA) * max(0, xB - xA)
            ###
            box1_area = box1[2] * box1[3]
            box2_area = box2[2] * box2[3]
            ###
            iou = inter / max(1e-10, inter, box1_area + box2_area - inter)
            assert (iou >= 0. and iou <= 1.)
            '''
            if (iou > 0.75) and not tag[j]:
                confs[j] = confs[j] * (1 - iou)
                tag[j] = True
            '''
            if iou > 0.5:
                confs[j] = 0.

####################################

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

####################################

offset_np = np.array([
[[  0,   0], [  0,  24], [  0,  48], [  0,  72], [  0,  96], [  0, 120], [  0, 144], [  0, 168], [  0, 192], [  0, 216], [  0, 240], [  0, 264]],
[[ 24,   0], [ 24,  24], [ 24,  48], [ 24,  72], [ 24,  96], [ 24, 120], [ 24, 144], [ 24, 168], [ 24, 192], [ 24, 216], [ 24, 240], [ 24, 264]],
[[ 48,   0], [ 48,  24], [ 48,  48], [ 48,  72], [ 48,  96], [ 48, 120], [ 48, 144], [ 48, 168], [ 48, 192], [ 48, 216], [ 48, 240], [ 48, 264]],
[[ 72,   0], [ 72,  24], [ 72,  48], [ 72,  72], [ 72,  96], [ 72, 120], [ 72, 144], [ 72, 168], [ 72, 192], [ 72, 216], [ 72, 240], [ 72, 264]],
[[ 96,   0], [ 96,  24], [ 96,  48], [ 96,  72], [ 96,  96], [ 96, 120], [ 96, 144], [ 96, 168], [ 96, 192], [ 96, 216], [ 96, 240], [ 96, 264]],
[[120,   0], [120,  24], [120,  48], [120,  72], [120,  96], [120, 120], [120, 144], [120, 168], [120, 192], [120, 216], [120, 240], [120, 264]],
[[144,   0], [144,  24], [144,  48], [144,  72], [144,  96], [144, 120], [144, 144], [144, 168], [144, 192], [144, 216], [144, 240], [144, 264]],
[[168,   0], [168,  24], [168,  48], [168,  72], [168,  96], [168, 120], [168, 144], [168, 168], [168, 192], [168, 216], [168, 240], [168, 264]],
[[192,   0], [192,  24], [192,  48], [192,  72], [192,  96], [192, 120], [192, 144], [192, 168], [192, 192], [192, 216], [192, 240], [192, 264]],
[[216,   0], [216,  24], [216,  48], [216,  72], [216,  96], [216, 120], [216, 144], [216, 168], [216, 192], [216, 216], [216, 240], [216, 264]]
])

kmeans = np.array([[ 47.938934,  35.145702],
                   [ 96.09451,   74.90686 ],
                   [ 29.959908,  22.899212],
                   [ 71.913376,  51.908134],
                   [ 15.042629,  41.93413 ],
                   [ 30.742947,  84.163376],
                   [133.14471,  112.522   ]])

def grid_to_pix(box):
    box[..., 0] = 24. * box[..., 0] + offset_np[..., 0] - 0.5 * box[..., 2]
    box[..., 1] = 24. * box[..., 1] + offset_np[..., 1] - 0.5 * box[..., 3]

    tmp = np.around(box[..., 0])
    box[..., 0] = np.around(box[..., 1])
    box[..., 1] = tmp

    tmp = np.around(box[..., 2])
    box[..., 2] = np.around(box[..., 3])
    box[..., 3] = tmp
    return box

####################################

def calc_map(id, truth, pred):

    truth = np.copy(truth)
    pred = np.copy(pred)

    print (np.shape(pred))
    print (np.shape(truth))

    # [12800, 8, 7, 5, 6, 8]
    truth[..., 2] = truth[..., 2] * np.reshape(kmeans[:, 1], (1, 1, 7, 1, 1))
    truth[..., 3] = truth[..., 3] * np.reshape(kmeans[:, 0], (1, 1, 7, 1, 1))
    # [12800, 7, 5, 6, 7]
    pred[..., 0] = sigmoid(pred[..., 0])
    pred[..., 1] = sigmoid(pred[..., 1])
    pred[..., 2] = np.exp(pred[..., 2]) * np.reshape(kmeans[:, 1], (1, 7, 1, 1))
    pred[..., 3] = np.exp(pred[..., 3]) * np.reshape(kmeans[:, 0], (1, 7, 1, 1))

    # nb, nd, ny, nx, nbox = np.shape(truth)
    # print (np.shape(truth)) # (16, 8, 5, 6, 8)
    # print (np.shape(pred))  # (16, 5, 6, 12)
    assert (len(truth) == len(pred))
    N = len(truth)

    truth_list = []
    det_list = []
    for n in range(N):

        #####################
        #####################
        #####################
        
        dets = []

        boxes   = grid_to_pix(truth[n, :, :, :, :, 0:4])
        objs    = truth[n, :, :, :, :, 4]
        no_objs = truth[n, :, :, :, :, 5]
        cats    = truth[n, :, :, :, :, 6]
        vld     = truth[n, :, :, :, :, 7]

        obj = np.where(truth[n, :, :, :, :, 4] == 1)
        box = boxes[obj]
        cat = cats[obj].astype(int)

        ndet = len(box)
        for d in range(ndet):
            det = (0,) + tuple(box[d]) + (cat[d],) + (1,) + (0,)
            # det = (0,) + tuple(box[d]) + (0,) + (1,) + (0,)
            dets.append(det)

        truth_n = np.concatenate((box, cat.reshape(-1, 1)), axis=1)
        truth_list.append(dets)

        #####################
        #####################
        #####################

        dets = []

        box  = grid_to_pix(pred[n, :, :, :, 0:4]).reshape(-1, 4)
        conf = pred[n, :, :, :, 4].reshape(-1)
        cat  = np.argmax(pred[n, :, :, :, 5:7], axis=-1).reshape(-1)

        order = np.argsort(conf)[::-1]
        box = box[order]
        conf = conf[order]
        cat = cat[order]

        #####################
        '''              
        soft_nms(box, conf)
        order = np.argsort(conf)[::-1]
        box = box[order]
        conf = conf[order]
        cat = cat[order]
        '''
        #####################

        ndet = len(box)
        for d in range(ndet):
            det = (0,) + tuple(box[d]) + (cat[d],) + (conf[d],) + (0,)
            # det = (0,) + tuple(box[d]) + (0,) + (conf[d],) + (0,)
            dets.append(det)

        pred_n = np.concatenate((box, conf.reshape(-1, 1), cat.reshape(-1, 1)), axis=1)
        det_list.append(dets)

        #####################
        #####################
        #####################
        '''
        src_image = plt.imread('./dataset/images/%d.jpg' % (id[n]))
        dst_image = './results/%d.jpg' % (id[n])
        draw_box(dst_image, src_image, truth_n, pred_n)
        '''
        #####################

    _coco_eval(truth_list, det_list, height=240, width=288)

####################################


















