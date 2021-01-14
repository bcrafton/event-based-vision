
import numpy as np
import tensorflow as tf

offset_np = [
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
]
offset = tf.constant(offset_np, dtype=tf.float32)

kmeans = np.array([[ 47.938934,  35.145702],
                   [ 96.09451,   74.90686 ],
                   [ 29.959908,  22.899212],
                   [ 71.913376,  51.908134],
                   [ 15.042629,  41.93413 ],
                   [ 30.742947,  84.163376],
                   [133.14471,  112.522   ]])
kmeans = tf.constant(kmeans, dtype=tf.float32)

@tf.function(experimental_relax_shapes=False)
def grid_to_pix(box):
    pix_box_yx = 24. * box[..., 0:2] + offset
    pix_box_h = box[..., 2:3] * tf.reshape(kmeans[:, 1], (1, 1, 7, 1, 1, 1))
    pix_box_w = box[..., 3:4] * tf.reshape(kmeans[:, 0], (1, 1, 7, 1, 1, 1))
    pix_box = tf.concat((pix_box_yx, pix_box_h, pix_box_w), axis=-1)
    return pix_box

@tf.function(experimental_relax_shapes=False)
def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = tf.maximum(boxA[...,0] - 0.5 * boxA[...,2], boxB[...,0] - 0.5 * boxB[...,2])
    yB = tf.minimum(boxA[...,0] + 0.5 * boxA[...,2], boxB[...,0] + 0.5 * boxB[...,2])

    xA = tf.maximum(boxA[...,1] - 0.5 * boxA[...,3], boxB[...,1] - 0.5 * boxB[...,3])
    xB = tf.minimum(boxA[...,1] + 0.5 * boxA[...,3], boxB[...,1] + 0.5 * boxB[...,3])

    # compute the area of intersection rectangle
    iy = yB - yA
    ix = xB - xA
    interArea = tf.maximum(tf.zeros_like(iy), iy) * tf.maximum(tf.zeros_like(ix), ix)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = tf.abs(boxA[...,2] * boxA[...,3])
    boxBArea = tf.abs(boxB[...,2] * boxB[...,3])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / tf.maximum(boxAArea + boxBArea - interArea, tf.maximum(tf.constant(1e-6), interArea))

    # return the intersection over union value
    return iou

@tf.custom_gradient
def abs_no_grad(x):
    def grad(dy):
        return dy
    return tf.abs(x), grad

@tf.custom_gradient
def sign_no_grad(x):
    def grad(dy):
        return dy
    return tf.sign(x), grad

@tf.function(experimental_relax_shapes=False)
def yolo_loss(batch_size, pred, label):

    # pred   = [4,     7, 7, 90]
    # label  = [4, -1, 7, 7, 5]
    # obj    = [4, -1, 7, 7]
    # no_obj = [4, -1, 7, 7]
    # cat    = [4, -1, 7, 7]

    pred   = tf.reshape(pred,  [batch_size, 1, 7, 10, 12, 7])

    label  = tf.reshape(label, [batch_size, 8, 7, 10, 12, 8])
    obj    = label[..., 4]
    no_obj = label[..., 5]
    cat    = tf.cast(label[..., 6], dtype=tf.int32)
    vld    = label[..., 7]

    ######################################

    label_box = grid_to_pix(label[..., 0:4])

    # pick the right one below!

    pred_box_hw = tf.exp(pred[..., 2:4])
    # pred_box_hw = pred[..., 2:4]

    # pred_box_yx = pred[..., 0:2]
    pred_box_yx = tf.sigmoid(pred[..., 0:2])

    pred_box = tf.concat((pred_box_yx, pred_box_hw), axis=-1)
    pred_box = grid_to_pix(pred_box)
    iou = calc_iou(pred_box, label_box)

    ############################

    label_yx = label[..., 0:2]
    pred_yx = pred[..., 0:2]

    ############################

    label_hw = label[..., 2:4]
    pred_hw = pred[..., 2:4]

    ############################

    label_conf = label[..., 4]
    pred_conf = pred[..., 4]

    ############################

    label_cat = tf.one_hot(cat, depth=2)
    pred_cat = pred[..., 5:7]

    ############################

    # tf.print(tf.reduce_sum(obj), tf.reduce_sum(vld))
    # tf.print(tf.shape(obj), tf.shape(vld), tf.shape(pred_yx), tf.shape(label_yx))
    # tf.print(tf.shape(no_obj), tf.shape(obj), tf.shape(iou), tf.shape(pred_conf))

    # yx_loss = 5. * obj * vld * tf.reduce_sum(tf.square(pred_yx - label_yx), axis=-1)
    yx_loss = 5. * obj * vld * tf.reduce_sum(tf.square(tf.sigmoid(pred_yx) - label_yx), axis=-1)
    yx_loss = tf.reduce_mean(tf.reduce_sum(yx_loss, axis=[2, 3, 4]))
    
    ######################################

    hw_loss = 5. * obj * vld * tf.reduce_sum(tf.square(tf.exp(pred_hw) - label_hw), axis=-1)
    # hw_loss = 5. * obj * vld * tf.reduce_sum(tf.square(pred_hw - label_hw), axis=-1)
    hw_loss = tf.reduce_mean(tf.reduce_sum(hw_loss, axis=[2, 3, 4]))

    ######################################
    
    # conf_loss = 1. * obj * vld * tf.square(pred_conf - 1.)
    conf_loss = 1. * obj * vld * tf.square(pred_conf - tf.stop_gradient(iou))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[2, 3, 4]))

    ######################################    
    
    # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    # none_conf = tf.expand_dims(pred_conf, axis=-1)
    # none_loss = bce(tf.zeros_like(none_conf), none_conf)
    # none_loss = 0.5 * no_obj * vld * none_loss
    # none_loss = tf.reduce_mean(tf.reduce_sum(none_loss, axis=[2, 3, 4]))

    none_loss = 0.5 * no_obj * vld * tf.square(pred_conf)
    none_loss = tf.reduce_mean(tf.reduce_sum(none_loss, axis=[2, 3, 4]))

    ######################################

    pred_cat = tf.repeat(pred_cat, 8, 1)
    cat_loss = obj * vld * tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=pred_cat)
    cat_loss = tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3, 4]))
    
    ######################################
    
    loss = yx_loss + hw_loss + conf_loss + none_loss + cat_loss
    return loss, (yx_loss, hw_loss, conf_loss, none_loss, cat_loss)









