
import numpy as np
import tensorflow as tf
from dataset.src.metrics.coco_eval import evaluate_detection

####################################

offset_np = np.array([
[  [0, 0],   [0, 48],   [0, 96],   [0, 144],   [0, 192],   [0, 240]], 
[ [48, 0],  [48, 48],  [48, 96],  [48, 144],  [48, 192],  [48, 240]], 
[ [96, 0],  [96, 48],  [96, 96],  [96, 144],  [96, 192],  [96, 240]], 
[[144, 0], [144, 48], [144, 96], [144, 144], [144, 192], [144, 240]], 
[[192, 0], [192, 48], [192, 96], [192, 144], [192, 192], [192, 240]]
])

def grid_to_pix(box):
    box[..., 2] = np.square(box[..., 2] * np.sqrt(240.))
    box[..., 3] = np.square(box[..., 3] * np.sqrt(288.))
    box[..., 0] = 48. * box[..., 0] + offset_np[..., 0] - 0.5 * box[..., 2]
    box[..., 1] = 48. * box[..., 1] + offset_np[..., 1] - 0.5 * box[..., 3]
    
    box_t = np.copy(box)
    box_t[..., 0] = np.around(box[..., 1])
    box_t[..., 1] = np.around(box[..., 0])
    box_t[..., 2] = np.around(box[..., 3])
    box_t[..., 3] = np.around(box[..., 2])
    return box_t

####################################

def calc_map(truth, pred):
    truth = truth[0:len(pred)]
    assert (len(truth) == len(pred))
    
    N = len(truth)
    dets = []
    for n in range(N):
        t = truth[n][0][0]
        cat = np.argmax(pred[n][:, :, 10:12], axis=-1)
        
        box1  = grid_to_pix(pred[n][:, :, 0:4])
        conf1 = pred[n][:, :, 4]
        obj1 = np.where(conf1 > 0)
        boxes1 = box1[obj1]
        conf1 = conf1[obj1]
        cat1 = cat[obj1]
        
        ndet = len(conf1)
        for d in range(ndet):
            det = (t,) + tuple(boxes1[d]) + (cat1[d],) + (conf1[d],) + (0,)
            dets.append(det)
        
        box2  = grid_to_pix(pred[n][:, :, 5:9])
        conf2 = pred[n][:, :, 9]
        obj2 = np.where(conf2 > 0)
        boxes2 = box2[obj2]
        conf2 = conf2[obj2]
        cat2 = cat[obj2]

        ndet = len(conf2)
        for d in range(ndet):
            det = (t,) + tuple(boxes2[d]) + (cat2[d],) + (conf2[d],) + (0,)
            dets.append(det)
        
    flat_truth = []
    for n in range(N):
        D = len(truth[n])
        for d in range(D):
            flat_truth.append(truth[n][d])
    
    header = [
    ('ts',         '<i8'), 
    ('x',          '<f4'),
    ('y',          '<f4'),
    ('w',          '<f4'),
    ('h',          '<f4'),
    ('class_id',   '<i8'),
    ('confidence', '<f4'),
    ('track_id',   '<i8')]
    
    dets = np.array(dets, dtype=header)
    flat_truth = np.array(flat_truth, dtype=header)
    evaluate_detection([dets], [flat_truth])
    
####################################
