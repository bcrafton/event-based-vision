
import numpy as np
import matplotlib.pyplot as plt
import cv2

##############################################################

offset_np = [
[  [0, 0],   [0, 48],   [0, 96],   [0, 144],   [0, 192],   [0, 240]], 
[ [48, 0],  [48, 48],  [48, 96],  [48, 144],  [48, 192],  [48, 240]], 
[ [96, 0],  [96, 48],  [96, 96],  [96, 144],  [96, 192],  [96, 240]], 
[[144, 0], [144, 48], [144, 96], [144, 144], [144, 192], [144, 240]], 
[[192, 0], [192, 48], [192, 96], [192, 144], [192, 192], [192, 240]]
]

def grid_to_pix(box):
    box[..., 0:2] = 48.  * box[..., 0:2] + offset_np
    box[..., 2]   = np.square(box[..., 2]) * 240.
    box[..., 3]   = np.square(box[..., 3]) * 288.
    return box

##############################################################

def calc_iou(label, pred1, pred2):
    iou1 = calc_iou_help(label, pred1)
    iou2 = calc_iou_help(label, pred2)
    return np.stack([iou1, iou2], 3)

def calc_iou_help(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = np.maximum(boxA[...,0] - 0.5 * boxA[...,2], boxB[...,0] - 0.5 * boxB[...,2])
    yB = np.minimum(boxA[...,0] + 0.5 * boxA[...,2], boxB[...,0] + 0.5 * boxB[...,2])

    xA = np.maximum(boxA[...,1] - 0.5 * boxA[...,3], boxB[...,1] - 0.5 * boxB[...,3])
    xB = np.minimum(boxA[...,1] + 0.5 * boxA[...,3], boxB[...,1] + 0.5 * boxB[...,3])

    # compute the area of intersection rectangle
    iy = yB - yA
    ix = xB - xA
    interArea = np.maximum(np.zeros_like(iy), iy) * np.maximum(np.zeros_like(ix), ix)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = np.absolute(boxA[...,2] * boxA[...,3])
    boxBArea = np.absolute(boxB[...,2] * boxB[...,3])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def draw_box(name, image, truth, pred):
    true_image = np.copy(image)
    pred_image = np.copy(image)

    ############################################

    boxes   = grid_to_pix(truth[:, :, :, 0:4])
    objs    = truth[:, :, :, 4]
    no_objs = truth[:, :, :, 5]
    cats    = truth[:, :, :, 6]
    vld     = truth[:, :, :, 7]
    
    obj = np.where(truth[:, :, :, 4] == 1)
    box = boxes[obj]
    cat = cats[obj]
    
    ndet = len(box)
    for d in range(ndet):
        draw_box_help(true_image, box[d], None)

    ############################################
    
    cat = np.argmax(pred[:, :, 10:12], axis=-1)
    
    box1  = grid_to_pix(pred[:, :, 0:4])
    conf1 = pred[:, :, 4]
    obj1 = np.where(conf1 > 0.25)
    boxes1 = box1[obj1]
    conf1 = conf1[obj1]
    cat1 = cat[obj1]
    
    ndet = len(conf1)
    for d in range(ndet):
        draw_box_help(pred_image, boxes1[d], None)
    
    ############################################

    cat = np.argmax(pred[:, :, 10:12], axis=-1)

    box2  = grid_to_pix(pred[:, :, 5:9])
    conf2 = pred[:, :, 9]
    obj2 = np.where(conf2 > 0.25)
    boxes2 = box2[obj2]
    conf2 = conf2[obj2]
    cat2 = cat[obj2]

    ndet = len(conf2)
    for d in range(ndet):
        draw_box_help(pred_image, boxes2[d], None)
    
    ############################################
        
    concat = np.concatenate((true_image, pred_image), axis=1)
    plt.imsave(name, concat)

def draw_box_help(image, box, color):
    [y, x, h, w] = box
    pt1 = (int(x-0.5*w), int(y-0.5*h))
    pt2 = (int(x+0.5*w), int(y+0.5*h))
    cv2.rectangle(image, pt1, pt2, 0, 1)










