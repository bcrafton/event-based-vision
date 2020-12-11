
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

def draw_box(name, image, truth, pred):
    true_image = np.copy(image)
    pred_image = np.copy(image)
    truth = np.copy(truth)
    pred = np.copy(pred)

    ############################################

    box = truth[:, 0:4]
    cat = truth[:, 4]
    
    ndet = len(box)
    for d in range(ndet):
        draw_box_help(true_image, box[d], cat[d], None)

    ############################################
    
    box = pred[:, 0:4]
    conf = pred[:, 4]
    cat = pred[:, 5]
    
    ndet = len(box)
    for d in range(ndet):
        if conf[d] > 0.25: draw_box_help(pred_image, box[d], cat[d], None)
    
    ############################################
        
    concat = np.concatenate((true_image, pred_image), axis=1)
    plt.imsave(name, concat, cmap='gray')

def draw_box_help(image, box, cat, color):
    [x, y, w, h] = box
    pt1 = (int(x), int(y))
    pt2 = (int(x+w), int(y+h))
    cv2.rectangle(image, pt1, pt2, 0, 1)
    label = 'Human' if cat == 1 else 'Car'
    cv2.putText(image, label, (int(x), int(y)), 0, 0.3, (0, 255, 0))









