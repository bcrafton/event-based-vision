
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
    pix_box_yx = 48. * box[:, :, :, :, 0:2] + offset
    pix_box_h = tf.square(box[:, :, :, :, 2:3]) * 240.
    pix_box_w = tf.square(box[:, :, :, :, 3:4]) * 288.
    pix_box = tf.concat((pix_box_yx, pix_box_h, pix_box_w), axis=4)
    return pix_box

@tf.function(experimental_relax_shapes=False)
def calc_iou(boxA, boxB, realBox):
    iou1 = calc_iou_help(boxA, realBox)
    iou2 = calc_iou_help(boxB, realBox)
    return tf.stack([iou1, iou2], axis=4)

@tf.function(experimental_relax_shapes=False)
def calc_iou_help(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = tf.maximum(boxA[:,:,:,:,0] - 0.5 * boxA[:,:,:,:,2], boxB[:,:,:,:,0] - 0.5 * boxB[:,:,:,:,2])
    yB = tf.minimum(boxA[:,:,:,:,0] + 0.5 * boxA[:,:,:,:,2], boxB[:,:,:,:,0] + 0.5 * boxB[:,:,:,:,2])

    xA = tf.maximum(boxA[:,:,:,:,1] - 0.5 * boxA[:,:,:,:,3], boxB[:,:,:,:,1] - 0.5 * boxB[:,:,:,:,3])
    xB = tf.minimum(boxA[:,:,:,:,1] + 0.5 * boxA[:,:,:,:,3], boxB[:,:,:,:,1] + 0.5 * boxB[:,:,:,:,3])

    # compute the area of intersection rectangle
    iy = yB - yA
    ix = xB - xA
    interArea = tf.maximum(tf.zeros_like(iy), iy) * tf.maximum(tf.zeros_like(ix), ix)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = abs_no_grad(boxA[:,:,:,:,2] * boxA[:,:,:,:,3])
    boxBArea = abs_no_grad(boxB[:,:,:,:,2] * boxB[:,:,:,:,3])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

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
def yolo_loss(label, pred):

    # pred   = [4,     7, 7, 90]
    # label  = [4, -1, 7, 7, 5]
    # obj    = [4, -1, 7, 7]
    # no_obj = [4, -1, 7, 7]
    # cat    = [4, -1, 7, 7]

    # TODO: DONT USE (-1) ANYWHERE ... this always results in problems.
    pred = tf.reshape(pred, [32, 1, 5, 6, 14])
    obj    = label[:, :, :, :, 4]
    no_obj = label[:, :, :, :, 5]
    cat    = tf.cast(label[:, :, :, :, 6], dtype=tf.int32)
    vld    = label[:, :, :, :, 7]

    ######################################

    label_box = grid_to_pix(label[:, :, :, :, 0:4])
    pred_box1 = grid_to_pix(pred[:, :, :, :, 0:4])
    pred_box2 = grid_to_pix(pred[:, :, :, :, 5:9])

    ############################

    label_yx = label[:, :, :, :, 0:2]
    pred_yx1 = pred[:, :, :, :, 0:2]
    pred_yx2 = pred[:, :, :, :, 5:7]

    ############################

    # label_hw = tf.sqrt(label[:, :, :, :, 2:4])
    # pred_hw1 = tf.sqrt(abs_no_grad(pred[:, :, :, :, 2:4])) * sign_no_grad(pred[:, :, :, :, 2:4])
    # pred_hw2 = tf.sqrt(abs_no_grad(pred[:, :, :, :, 7:9])) * sign_no_grad(pred[:, :, :, :, 7:9])

    label_hw = label[:, :, :, :, 2:4]
    pred_hw1 = pred[:, :, :, :, 2:4]
    pred_hw2 = pred[:, :, :, :, 7:9]

    ############################

    label_conf = label[:, :, :, :, 4]
    pred_conf1 = pred[:, :, :, :, 4]
    pred_conf2 = pred[:, :, :, :, 9]

    ############################

    label_cat = tf.one_hot(cat, depth=2)
    pred_cat1 = pred[:, :, :, :, 10:12]
    pred_cat2 = pred[:, :, :, :, 12:14]

    ############################

    iou = calc_iou(pred_box1, pred_box2, label_box)
    # resp_box = tf.greater(iou[:, :, :, :, 0], iou[:, :, :, :, 1])
    # pretty sure less/greater->bool is ideal because we use this thing in tf.where below
    # resp_box = tf.less(iou[:, :, :, :, 0], iou[:, :, :, :, 1])
    # we are a moron. tf.where -> if resp_box: 0 else: 1 ...
    resp_box = tf.greater(iou[:, :, :, :, 0], iou[:, :, :, :, 1])

    obj_mask1 = obj * tf.cast(tf.greater_equal(iou[:, :, :, :, 0], iou[:, :, :, :, 1]), tf.float32)
    obj_mask2 = obj * tf.cast(tf.greater      (iou[:, :, :, :, 1], iou[:, :, :, :, 0]), tf.float32)

    no_obj_mask1 = 1 - obj_mask1
    no_obj_mask2 = 1 - obj_mask2

    ######################################

    loss_yx1 = tf.reduce_sum(tf.square(pred_yx1 - label_yx), axis=4)
    loss_yx2 = tf.reduce_sum(tf.square(pred_yx2 - label_yx), axis=4)
    yx_loss = 5. * obj * vld * tf.where(resp_box, loss_yx1, loss_yx2)
    yx_loss = tf.reduce_mean(tf.reduce_sum(yx_loss, axis=[2, 3]))

    ######################################

    loss_hw1 = tf.reduce_sum(tf.square(pred_hw1 - label_hw), axis=4)
    loss_hw2 = tf.reduce_sum(tf.square(pred_hw2 - label_hw), axis=4)
    hw_loss = 5. * obj * vld * tf.where(resp_box, loss_hw1, loss_hw2)
    hw_loss = tf.reduce_mean(tf.reduce_sum(hw_loss, axis=[2, 3]))

    ######################################

    '''
    loss_obj1 = tf.square(pred_conf1 - label_conf)
    loss_obj2 = tf.square(pred_conf2 - label_conf)
    obj_loss = 1. * obj * vld * tf.where(resp_box, loss_obj1, loss_obj2)
    '''

    loss_obj1 = tf.square(pred_conf1 - 1) * obj_mask1
    loss_obj2 = tf.square(pred_conf2 - 1) * obj_mask2
    # obj_loss = 1. * obj * vld * (loss_obj1 + loss_obj2)
    obj_loss = 1. * vld * (loss_obj1 + loss_obj2)
    obj_loss = tf.reduce_mean(tf.reduce_sum(obj_loss, axis=[2, 3]))

    ######################################    

    '''
    loss_no_obj1 = tf.square(pred_conf1 - label_conf)
    loss_no_obj2 = tf.square(pred_conf2 - label_conf)
    no_obj_loss = 0.5 * no_obj * vld * tf.where(resp_box, loss_no_obj1, loss_no_obj2)
    '''

    loss_no_obj1 = tf.square(pred_conf1) * no_obj_mask1
    loss_no_obj2 = tf.square(pred_conf2) * no_obj_mask2
    # no_obj_loss = 0.5 * no_obj * vld * (loss_no_obj1 + loss_no_obj2)
    no_obj_loss = 0.5 * vld * (loss_no_obj1 + loss_no_obj2)
    no_obj_loss = tf.reduce_mean(tf.reduce_sum(no_obj_loss, axis=[2, 3]))

    ######################################

    # (1)
    pred_cat1 = tf.repeat(pred_cat1, 8, 1)
    pred_cat2 = tf.repeat(pred_cat2, 8, 1)

    cat_loss1 = obj * vld * tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=pred_cat1)
    cat_loss2 = obj * vld * tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=pred_cat2)
    cat_loss = tf.where(resp_box, cat_loss1, cat_loss2)

    cat_loss = tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3]))

    # (2)
    '''
    cat_loss = 2. * obj * vld * tf.reduce_sum(tf.square(pred_cat - label_cat), axis=4)
    cat_loss = tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3]))
    '''

    # (3)
    '''
    pred_cat1 = tf.repeat(pred_cat1, 8, 1)
    pred_cat2 = tf.repeat(pred_cat2, 8, 1)

    cat_loss1 = 2. * obj * vld * tf.reduce_sum(tf.constant([1, 10], dtype=tf.float32) * tf.square(pred_cat1 - label_cat), axis=4)
    cat_loss2 = 2. * obj * vld * tf.reduce_sum(tf.constant([1, 10], dtype=tf.float32) * tf.square(pred_cat2 - label_cat), axis=4)
    cat_loss = tf.where(resp_box, cat_loss1, cat_loss2)

    cat_loss = tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3]))
    '''

    # 228,123 cars and 27,658 pedestrians bounding boxes
    # (4)
    # lol - this is wrong.
    '''
    pred_cat1 = tf.repeat(pred_cat1, 8, 1)
    pred_cat2 = tf.repeat(pred_cat2, 8, 1)

    car_loss1 = obj * vld * tf.square(pred_cat1[..., 0] - label_cat[..., 0])
    ped_loss1 = obj * vld * tf.square(pred_cat1[..., 1] - label_cat[..., 1]) * 10
    cat_loss1 = car_loss1 + ped_loss1

    car_loss2 = obj * vld * tf.square(pred_cat2[..., 0] - label_cat[..., 0])
    ped_loss2 = obj * vld * tf.square(pred_cat2[..., 1] - label_cat[..., 1]) * 10
    cat_loss2 = car_loss2 + ped_loss2

    cat_loss = tf.where(resp_box, cat_loss1, cat_loss2)
    cat_loss = tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3]))
    '''

    # (5)
    '''
    pred_cat1 = tf.repeat(pred_cat1, 8, 1)
    pred_cat2 = tf.repeat(pred_cat2, 8, 1)

    cat_loss1 = obj * vld * tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=pred_cat1)
    cat_loss2 = obj * vld * tf.nn.softmax_cross_entropy_with_logits(labels=label_cat, logits=pred_cat2)

    cat_loss = tf.where(resp_box, cat_loss1, cat_loss2)
    cat_loss = tf.where(tf.equal(cat, tf.constant(1)), cat_loss * 10, cat_loss)

    cat_loss = 2 * tf.reduce_mean(tf.reduce_sum(cat_loss, axis=[2, 3]))
    '''

    ######################################

    # print (yx_loss.numpy(), hw_loss.numpy(), obj_loss.numpy(), no_obj_loss.numpy(), cat_loss.numpy())
    loss = yx_loss + hw_loss + obj_loss + no_obj_loss + cat_loss

    tf.print(tf.math.round(100. * yx_loss     / loss), 
             tf.math.round(100. * hw_loss     / loss),
             tf.math.round(100. * obj_loss    / loss),
             tf.math.round(100. * no_obj_loss / loss),
             tf.math.round(100. * cat_loss    / loss))

    return loss 









