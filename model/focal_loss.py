import tensorflow as tf

def focal_loss_sigmoid(target_tensor, prediction_tensor, alpha=0.25, gamma=2):
    per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor)
    prediction_probabilities = tf.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (target_tensor * alpha + (1 - target_tensor) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
    focal_cross_entropy_loss = tf.reduce_sum(focal_cross_entropy_loss)
    return focal_cross_entropy_loss


def focal_loss_with_reduce(target_tensor, prediction_tensor):
    loss = focal_loss_sigmoid(target_tensor, prediction_tensor)
    return tf.reduce_sum(loss, axis=(0,1,2)) / (tf.reduce_sum(target_tensor, axis=(0,1,2))+1)
