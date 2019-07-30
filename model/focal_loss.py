import tensorflow as tf

def focal_loss_sigmoid(pred,
                       target,
                       gamma=2.0,
                       alpha=0.25,
                       weight=None):
    # predict probability
    pred_sigmoid = tf.sigmoid(pred)

    # focal weight
    pt = tf.where(
        tf.equal(target, 1.0),
        1.0 - pred_sigmoid,
        pred_sigmoid)
    alpha_weight = (alpha * target) + ((1 - alpha) * (1 - target)) 
    focal_weight = alpha_weight * tf.pow(pt, gamma)
    if weight is not None:
        focal_weight = focal_weight*weight

    # loss
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred)
    loss = bce * focal_weight
    loss = tf.reduce_sum(loss)
    
    return loss

