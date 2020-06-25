
import numpy as np
import matplotlib.pyplot as plt

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
[ [96, 0],  [96, 48],  [96, 96],  [96, 144], [128, 192], [128, 240]], 
[[144, 0], [144, 48], [144, 96], [144, 144], [144, 192], [144, 240]], 
[[192, 0], [192, 48], [192, 96], [192, 144], [192, 192], [192, 240]]
]

def grid_to_pix(box):
    box[..., 0:2] = 48.  * box[..., 0:2] + offset_np
    box[..., 2:3] = 288. * box[..., 2:3]
    box[..., 3:4] = 240. * box[..., 3:4]
    return box

##############################################################

def calc_iou(label, pred1, pred2):
    iou1 = calc_iou_help(label, pred1)
    iou2 = calc_iou_help(label, pred2)
    return np.stack([iou1, iou2], 3)

def calc_iou_help(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[...,0] - 0.5 * boxA[...,2], boxB[...,0] - 0.5 * boxA[...,2])
    xB = np.minimum(boxA[...,0] + 0.5 * boxA[...,2], boxB[...,0] + 0.5 * boxB[...,2])

    yA = np.maximum(boxA[...,1] - 0.5 * boxA[...,3], boxB[...,1] - 0.5 * boxB[...,3])
    yB = np.minimum(boxA[...,1] + 0.5 * boxA[...,3], boxB[...,1] + 0.5 * boxB[...,3])

    # compute the area of intersection rectangle
    ix = xB - xA
    iy = yB - yA
    interArea = np.maximum(np.zeros_like(ix), ix) * np.maximum(np.zeros_like(iy), iy)

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
    # print (np.shape(label))
    objs      = label[..., 4]
    # print (np.shape(objs))
    true_boxs = grid_to_pix(label[..., 0:4])
    boxs1     = grid_to_pix(pred[..., 0:4])
    boxs2     = grid_to_pix(pred[..., 5:9])
    iou = calc_iou(true_boxs, boxs1, boxs2)

    true_image = np.copy(image)
    pred_image = np.copy(image)

    nbox = min(nbox, len(colors))
    for b in range(nbox):
        obj = objs[b]
        [xc, yc] = np.squeeze(np.where(obj > 0))

        box = np.array(true_boxs[b][xc][yc], dtype=int)
        draw_box_help(true_image, box, colors[b])
        
        iou1 = iou[b][xc][yc][0]
        iou2 = iou[b][xc][yc][1]
        if iou1 > iou2:
            box = np.array(boxs1[xc][yc], dtype=int)
        else:
            box = np.array(boxs2[xc][yc], dtype=int)
        draw_box_help(pred_image, box, colors[b])
        
    concat = np.concatenate((true_image, pred_image), axis=1)
    plt.imsave(name, concat)

def draw_box_help(image, box, color):
    [x, y, w, h] = box
    [x11, x12, x21, x22] = np.array([x-0.5*w, x-0.5*w+5, x+0.5*w-5, x+0.5*w], dtype=int)
    [y11, y12, y21, y22] = np.array([y-0.5*h, y-0.5*h+5, y+0.5*h-5, y+0.5*h], dtype=int)
    # image[y11:y12, x12:x21] = 1.
    # image[y21:y22, x12:x21] = 1.
    # image[y12:y21, x11:x12] = 1.
    # image[y12:y21, x21:x22] = 1.
    image[x11:x12, y12:y21] = 1.
    image[x21:x22, y12:y21] = 1.
    image[x12:x21, y11:y12] = 1.
    image[x12:x21, y21:y22] = 1.

##############################################################
'''
results_filename = 'yolo_coco.npy'
results = np.load(results_filename, allow_pickle=True).item()

##############################################################

for batch in range(100):
    img_name = 'img%d' % (batch)
    imgs = results[img_name]

    pred_name = 'pred%d' % (batch)
    preds = results[pred_name]

    label_name = 'label%d' % (batch)
    labels = results[label_name]
    coords, objs, no_objs, cats, vlds = labels

    #############

    for ex in range(8):
        image = imgs[ex]; image = image / np.max(image)
        label = coords[ex] * np.expand_dims(vlds[ex], axis=3)
        pred = preds[ex]
        nbox = np.count_nonzero(np.average(vlds[ex], axis=(1,2)))
        draw_box('./results/img%d.jpg' % (batch * 8 + ex), image, label, pred, nbox)
'''
##############################################################












