import tensorflow as tf

def iou_loss(pred, target, weight=None):
    '''
    Args:
        pred: [n, 4]
        target: [n, 4]
        weight: [n]
    Returns:
        loss: average iou loss
    '''
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]
    pred = tf.reduce_min(pred, 1)
    # print("pred", pred[pred<0])

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]
    target = tf.reduce_min(target, 1)
    # print("target", target[target<0])

    target_area = (target_left + target_right) * \
                    (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = tf.minimum(pred_left, target_left) + \
                    tf.minimum(pred_right, target_right)
    h_intersect = tf.minimum(pred_bottom, target_bottom) + \
                    tf.minimum(pred_top, target_top)

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    #print(tf.reduce_sum(tf.cast(area_union<=0, tf.float32) ))
    #print(tf.reduce_sum(tf.cast(area_intersect<=0, tf.float32) ))
    ratio = (area_intersect + 1.0) / (area_union + 1.0)
    #print(tf.reduce_max(ratio), tf.reduce_min(ratio))

    losses = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))

    if weight is not None:
        return tf.reduce_sum(losses*weight) / tf.reduce_sum(weight)
    else:
        return tf.reduce_mean(losses)
