import tensorflow as tf

from model.resnet import resnet50_block
from model.fpn import fpn_block
from model.fcos_head import fcos_block


def build_fcos_detector(num_classes):
    inputs = tf.keras.Input(shape=(None, None, 3), dtype=tf.float32)
    x = resnet50_block(inputs, num_pyramid=3)
    x = fpn_block(x, fpn_dim=256, num_extra=2, num_group=32)
    outputs = fcos_block(x, conv_dim=256, num_classes=num_classes)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights("./data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
    model.summary()
    return model
