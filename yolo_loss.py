
import numpy as np
import tensorflow as tf

offset_np = [
[[  0,   0], [  0,  32], [  0,  64], [  0,  96], [  0, 128], [  0, 160], [  0, 192], [  0, 224], [  0, 256], [  0, 288], [  0, 320], [  0, 352], [  0, 384], [  0, 416]],
[[ 32,   0], [ 32,  32], [ 32,  64], [ 32,  96], [ 32, 128], [ 32, 160], [ 32, 192], [ 32, 224], [ 32, 256], [ 32, 288], [ 32, 320], [ 32, 352], [ 32, 384], [ 32, 416]],
[[ 64,   0], [ 64,  32], [ 64,  64], [ 64,  96], [ 64, 128], [ 64, 160], [ 64, 192], [ 64, 224], [ 64, 256], [ 64, 288], [ 64, 320], [ 64, 352], [ 64, 384], [ 64, 416]],
[[ 96,   0], [ 96,  32], [ 96,  64], [ 96,  96], [ 96, 128], [ 96, 160], [ 96, 192], [ 96, 224], [ 96, 256], [ 96, 288], [ 96, 320], [ 96, 352], [ 96, 384], [ 96, 416]],
[[128,   0], [128,  32], [128,  64], [128,  96], [128, 128], [128, 160], [128, 192], [128, 224], [128, 256], [128, 288], [128, 320], [128, 352], [128, 384], [128, 416]],
[[160,   0], [160,  32], [160,  64], [160,  96], [160, 128], [160, 160], [160, 192], [160, 224], [160, 256], [160, 288], [160, 320], [160, 352], [160, 384], [160, 416]],
[[192,   0], [192,  32], [192,  64], [192,  96], [192, 128], [192, 160], [192, 192], [192, 224], [192, 256], [192, 288], [192, 320], [192, 352], [192, 384], [192, 416]],
[[224,   0], [224,  32], [224,  64], [224,  96], [224, 128], [224, 160], [224, 192], [224, 224], [224, 256], [224, 288], [224, 320], [224, 352], [224, 384], [224, 416]]
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
    pix_box_yx = 32. * box[..., 0:2] + offset
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

    pred   = tf.reshape(pred,  [batch_size, 1, 7, 8, 14, 12])

    label  = tf.reshape(label, [batch_size, 8, 7, 8, 14, 8])
    obj    = label[..., 4]
    no_obj = label[..., 5]
    cat    = tf.cast(label[..., 6], dtype=tf.int32)
    vld    = label[..., 7]

    ######################################

    label_box = grid_to_pix(label[..., 0:4])
    pred_box_hw = tf.exp(pred[..., 2:4])
    pred_box_yx =        pred[..., 0:2]
    pred_box = tf.concat((pred_box_yx, pred_box_hw), axis=-1)
    pred_box = grid_to_pix(pred_box)

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

    label_cat = tf.one_hot(cat, depth=7)
    pred_cat = pred[..., 5:12]

    ############################

    iou = calc_iou(pred_box, label_box)
    # tf.print (tf.shape(iou), tf.shape(obj))

    ######################################

    yx_loss = 5. * obj * vld * tf.reduce_sum(tf.square(pred_yx - label_yx), axis=-1)    
    # yx_loss = tf.transpose(yx_loss, (0, 1, 3, 4, 2))
    # yx_loss = tf.gather(yx_loss, resp_box, axis=4, batch_dims=4)
    yx_loss = tf.reduce_mean(tf.reduce_sum(yx_loss, axis=[2, 3, 4]))
    
    ######################################

    # the problem might be tf.exp() blows up.
    # especially when not loading weights.
    # hw_loss = 5. * obj * vld * tf.reduce_sum(tf.square(tf.exp(pred_hw) - label_hw), axis=-1)
    hw_loss = 5. * obj * vld * tf.reduce_sum(tf.square(pred_hw - label_hw), axis=-1)
    
    # hw_loss = tf.transpose(hw_loss, (0, 1, 3, 4, 2))
    # hw_loss = tf.gather(hw_loss, resp_box, axis=4, batch_dims=4)
    hw_loss = tf.reduce_mean(tf.reduce_sum(hw_loss, axis=[2, 3, 4]))

    ######################################

    '''
    conf_loss = tf.transpose(1. * vld * pred_conf, (0, 1, 3, 4, 2))
    conf_loss = tf.square(conf_loss - tf.stop_gradient(iou))
    conf_loss = tf.gather(conf_loss, resp_box, axis=4, batch_dims=4)
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[2, 3]))
    '''
    
    # conf_loss = 1. * obj * vld * tf.square(pred_conf - 1.)
    conf_loss = 1. * obj * vld * tf.square(pred_conf - tf.stop_gradient(iou))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[2, 3, 4]))

    ######################################    
    
    '''
    none_loss = tf.transpose(0.5 * vld * pred_conf, (0, 1, 3, 4, 2))
    none_loss = tf.square(none_loss)
    none_loss = tf.gather(none_loss, resp_box, axis=4, batch_dims=4)
    none_loss = tf.reduce_mean(tf.reduce_sum(none_loss, axis=[2, 3]))
    '''
    
    none_loss = 0.5 * no_obj * vld * tf.square(pred_conf)
    none_loss = tf.reduce_mean(tf.reduce_sum(none_loss, axis=[2, 3, 4]))

    ######################################

    pred_cat = tf.repeat(pred_cat, 8, 1)
    cat_loss = obj * vld * tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=pred_cat)
    # cat_loss = tf.transpose(cat_loss, (0, 1, 3, 4, 2))
    # cat_loss = tf.gather(cat_loss, resp_box, axis=4, batch_dims=4)
    cat_loss = tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3, 4]))
    
    ######################################
    
    loss = yx_loss + hw_loss + conf_loss + none_loss + cat_loss
    return loss, (yx_loss, hw_loss, conf_loss, none_loss, cat_loss)









