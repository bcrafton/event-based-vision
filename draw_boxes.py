
import numpy as np
import matplotlib.pyplot as plt
import cv2

##############################################################

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

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

##############################################################

def draw_box(name, image, truth, pred):

    image = np.stack((image, image, image), axis=2)
    image = image / np.max(image)

    ############################################
    
    true_image = np.copy(image)
    pred_image = np.copy(image)
    truth = np.copy(truth)
    pred = np.copy(pred)

    ############################################

    truth[..., 2] = truth[..., 2] * np.reshape(kmeans[:, 1], (7, 1, 1))
    truth[..., 3] = truth[..., 3] * np.reshape(kmeans[:, 0], (7, 1, 1))

    pred[..., 0] = sigmoid(pred[..., 0])
    pred[..., 1] = sigmoid(pred[..., 1])
    pred[..., 2] = np.exp(pred[..., 2]) * np.reshape(kmeans[:, 1], (7, 1, 1))
    pred[..., 3] = np.exp(pred[..., 3]) * np.reshape(kmeans[:, 0], (7, 1, 1))

    ############################################

    dets = []

    boxes   = grid_to_pix(truth[:, :, :, :, 0:4])
    objs    = truth[:, :, :, :, 4]
    no_objs = truth[:, :, :, :, 5]
    cats    = truth[:, :, :, :, 6]
    vld     = truth[:, :, :, :, 7]

    obj = np.where(truth[:, :, :, :, 4] == 1)
    box = boxes[obj]
    cat = cats[obj].astype(int)

    ndet = len(box)
    for d in range(ndet):
        true_image = draw_box_help(true_image, box[d], cat[d], color=(1,0,0))

    ############################################

    dets = []

    box  = grid_to_pix(pred[:, :, :, 0:4]).reshape(-1, 4)
    conf = pred[:, :, :, 4].reshape(-1)
    cat  = np.argmax(pred[:, :, :, 5:7], axis=-1).reshape(-1)

    order = np.argsort(conf)[::-1]
    box = box[order]
    conf = conf[order]
    cat = cat[order]

    for d in range(ndet):
        pred_image = draw_box_help(pred_image, box[d], cat[d], color=(0,0,1))

    ############################################

    H, W, N = np.shape(true_image)
    mid = np.zeros(shape=(H, 5, N))
    concat = np.concatenate((true_image, mid, pred_image), axis=1)
    plt.imsave(name, concat, dpi=300)

    ############################################

def draw_box_help(image, box, cat, color):
    [x, y, w, h] = box
    pt1 = (int(x), int(y))
    pt2 = (int(x+w), int(y+h))
    image = cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color, thickness=2)
    return image








