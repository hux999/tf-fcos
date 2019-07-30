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

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

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

    ratio = (area_intersect + 1.0) / (area_union + 1.0)

    losses = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))

    if weight is not None:
        return tf.reduce_sum(losses*weight) / tf.reduce_sum(weight)
    else:
        return tf.reduce_mean(losses)
