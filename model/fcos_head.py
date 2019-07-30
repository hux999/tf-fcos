import math

import tensorflow as tf
from tensorflow.keras import layers, Sequential

from layers.group_normalization import GroupNormalization
from layers.scale_layer import ScaleLayer


class FCOSHead(tf.keras.Model):

    def __init__(self, conv_dim, num_classes, prefix="fcos_head"):
        super(FCOSHead, self).__init__()
        self.cls_tower = Sequential(name=(prefix+"_cls_tower"))
        self.bbox_tower = Sequential(name=(prefix+"_bbox_tower"))
        for _ in range(4):
            self.cls_tower.add(
                layers.Conv2D(
                    conv_dim,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                )
            )
            self.cls_tower.add(GroupNormalization(groups=32))
            self.cls_tower.add(layers.ReLU())
            self.bbox_tower.add(
                layers.Conv2D(
                    conv_dim,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                )
            )
            self.bbox_tower.add(GroupNormalization(groups=32))
            self.bbox_tower.add(layers.ReLU())
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_logits = layers.Conv2D(
            num_classes,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Constant(bias_value),
            name=(prefix+"_cls_logits")
            )
        self.bbox_pred = layers.Conv2D(
            4,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name=(prefix+"_bbox_pred")
            )
        self.centerness = layers.Conv2D(
            1,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name=(prefix+"_centerness")
            )
       
    def call(self, inputs):
        cls_feature = self.cls_tower(inputs)
        bbox_feature = self.bbox_tower(inputs)
        logits = self.cls_logits(cls_feature)
        centerness = self.centerness(cls_feature)
        bbox_reg = self.bbox_pred(bbox_feature)
        return logits, bbox_reg, centerness


def fcos_block(inputs, conv_dim, num_classes, prefix="fcos_head"):
    logits_list = []
    bbox_reg_list = []
    centerness_list = []
    num_pyramid = len(inputs)
    fcos_head = FCOSHead(conv_dim, num_classes, prefix)
    for i in range(num_pyramid):
        logits, bbox_reg, centerness = fcos_head(inputs[i])
        name = prefix + ("_scale%d" % i)
        scale_bbox_reg = ScaleLayer(name=name)(bbox_reg)
        scale_bbox_reg = tf.math.exp(scale_bbox_reg)
        logits_list.append(logits)
        bbox_reg_list.append(scale_bbox_reg)
        centerness_list.append(centerness)
    return [logits_list, bbox_reg_list, centerness_list]

def compute_locations(feature, stride, dim_height=1, dim_width=2):
    # type: ( Tensor, int, int, float ) -> Tensor
    w = tf.cast(tf.shape(feature)[dim_width], tf.float32)
    h = tf.cast(tf.shape(feature)[dim_height], tf.float32)

    shifts_x = tf.range(0, w * stride, stride, dtype=tf.float32)
    shifts_y = tf.range(0, h * stride, stride, dtype=tf.float32)

    shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
    shift_x = tf.reshape(shift_x, [-1])
    shift_y = tf.reshape(shift_y, [-1])
    locations = tf.stack([shift_x, shift_y], -1) + stride // 2
    return locations