
import numpy as np
import matplotlib.pyplot as plt
import cv2

##############################################################

colors = [
np.array([1.0, 0.0, 0.0]),
np.array([0.0, 1.0, 0.0]),
np.array([0.0, 0.0, 1.0]),

np.array([1.0, 1.0, 0.0]),
np.array([1.0, 0.0, 1.0]),
np.array([0.0, 1.0, 1.0])
]

color_names = [
'red',
'green',
'blue',
'yellow',
'violet',
'cyan'
]

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

def draw_box(name, image, label, pred, nbox):
    objs      = label[..., 4]
    true_boxs = grid_to_pix(label[..., 0:4])
    boxs1     = grid_to_pix(pred[..., 0:4])
    boxs2     = grid_to_pix(pred[..., 5:9])
    iou = calc_iou(true_boxs, boxs1, boxs2)

    conf1 = pred[..., 4]
    conf2 = pred[..., 9]

    true_image = np.copy(image)
    pred_image = np.copy(image)

    nbox = min(nbox, len(colors))
    for b in range(nbox):
        obj = objs[b]
        [yc, xc] = np.squeeze(np.where(obj > 0))

        box = np.array(true_boxs[b][yc][xc], dtype=int)
        draw_box_help(true_image, box, colors[b])
        
        '''
        iou1 = iou[b][yc][xc][0]
        iou2 = iou[b][yc][xc][1]
        
        conf1 = pred[yc][xc][4]
        conf2 = pred[yc][xc][9]
        
        # if conf1 > conf2:
        if iou1 > iou2:
            box = np.array(boxs1[yc][xc], dtype=int)
        else:
            box = np.array(boxs2[yc][xc], dtype=int)
        
        draw_box_help(pred_image, box, colors[b])
        '''
        
    for yc in range (5):
        for xc in range(6):
            if conf1[yc][xc] > 0.25:
                draw_box_help(pred_image, boxs1[yc][xc], None)
            if conf2[yc][xc] > 0.25:
                draw_box_help(pred_image, boxs2[yc][xc], None)
        
    concat = np.concatenate((true_image, pred_image), axis=1)
    plt.imsave(name, concat)

def draw_box_help(image, box, color):
    '''
    [y, x, h, w] = box
    [x11, x12, x21, x22] = np.array([x-0.5*w, x-0.5*w+5, x+0.5*w-5, x+0.5*w], dtype=int)
    [y11, y12, y21, y22] = np.array([y-0.5*h, y-0.5*h+5, y+0.5*h-5, y+0.5*h], dtype=int)
    image[y11:y12, x12:x21] = 1.
    image[y21:y22, x12:x21] = 1.
    image[y12:y21, x11:x12] = 1.
    image[y12:y21, x21:x22] = 1.
    '''
    '''
    pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
    pt2 = (pt1[0] + size[0], pt1[1] + size[1])
    cv2.rectangle(img, pt1, pt2, color, 1)
    '''
    [y, x, h, w] = box
    pt1 = (int(x-0.5*w), int(y-0.5*h))
    pt2 = (int(x+0.5*w), int(y+0.5*h))
    cv2.rectangle(image, pt1, pt2, 0, 1)










