
import tensorflow as tf

offset_np = [
[  [0, 0],   [0, 48],   [0, 96],   [0, 144],   [0, 192],   [0, 240]], 
[ [48, 0],  [48, 48],  [48, 96],  [48, 144],  [48, 192],  [48, 240]], 
[ [96, 0],  [96, 48],  [96, 96],  [96, 144],  [96, 192],  [96, 240]], 
[[144, 0], [144, 48], [144, 96], [144, 144], [144, 192], [144, 240]], 
[[192, 0], [192, 48], [192, 96], [192, 144], [192, 192], [192, 240]]
]

offset = tf.constant(offset_np, dtype=tf.float32)

@tf.function(experimental_relax_shapes=False)
def grid_to_pix(box):
    pix_box_yx = 48. * box[..., 0:2] + offset
    pix_box_h = tf.square(box[..., 2:3]) * 240.
    pix_box_w = tf.square(box[..., 3:4]) * 288.
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

    pred   = tf.reshape(pred,  [batch_size, 1, 5, 5, 6, 7])
    label  = tf.reshape(label, [batch_size, 8, 1, 5, 6, 8])
    obj    = label[..., 4]
    no_obj = label[..., 5]
    cat    = tf.cast(label[..., 6], dtype=tf.int32)
    vld    = label[..., 7]

    ######################################

    label_box = grid_to_pix(label[..., 0:4])
    pred_box = grid_to_pix(pred[..., 0:4])

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

    iou = calc_iou(pred_box, label_box)
    iou = tf.transpose(iou, (0, 1, 3, 4, 2))
    resp_box = tf.math.argmax(iou, axis=4)

    ######################################

    yx_loss = 5. * obj * vld * tf.reduce_sum(tf.square(pred_yx - label_yx), axis=-1)    
    yx_loss = tf.transpose(yx_loss, (0, 1, 3, 4, 2))
    yx_loss = tf.gather(yx_loss, resp_box, axis=4, batch_dims=4)
    yx_loss = tf.reduce_mean(tf.reduce_sum(yx_loss, axis=[2, 3]))
    
    ######################################

    hw_loss = 5. * obj * vld * tf.reduce_sum(tf.square(pred_hw - label_hw), axis=-1)    
    hw_loss = tf.transpose(hw_loss, (0, 1, 3, 4, 2))
    hw_loss = tf.gather(hw_loss, resp_box, axis=4, batch_dims=4)
    hw_loss = tf.reduce_mean(tf.reduce_sum(hw_loss, axis=[2, 3]))

    ######################################

    conf_loss = tf.transpose(1. * vld * pred_conf, (0, 1, 3, 4, 2))
    conf_loss = tf.square(conf_loss - tf.stop_gradient(iou))
    conf_loss = tf.gather(conf_loss, resp_box, axis=4, batch_dims=4)
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[2, 3]))

    ######################################    

    none_loss = tf.transpose(0.5 * vld * pred_conf, (0, 1, 3, 4, 2))
    none_loss = tf.square(none_loss)
    none_loss = tf.gather(none_loss, resp_box, axis=4, batch_dims=4)
    none_loss = tf.reduce_mean(tf.reduce_sum(none_loss, axis=[2, 3]))

    ######################################

    pred_cat = tf.repeat(pred_cat, 8, 1)
    label_cat = tf.repeat(label_cat, 5, 2)
    cat_loss = obj * vld * tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=pred_cat)
    cat_loss = tf.transpose(cat_loss, (0, 1, 3, 4, 2))
    cat_loss = tf.gather(cat_loss, resp_box, axis=4, batch_dims=4)
    cat_loss = tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3]))
    
    ######################################
    
    loss = yx_loss + hw_loss + conf_loss + none_loss + cat_loss
    return loss, (yx_loss, hw_loss, conf_loss, none_loss, cat_loss)









