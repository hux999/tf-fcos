import tensorflow as tf
from tensorflow.keras import layers
from layers.group_normalization import GroupNormalization


def build_latlayer(inputs, fpn_dim, num_group, prefix):
    x = layers.Conv2D(
        fpn_dim, 1, padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(),
        name=(prefix+"_conv"))(inputs)
    x = GroupNormalization(groups=num_group, name=(prefix+"_gn"))(x)
    return x

def build_smooth_layer(inputs, fpn_dim, num_group, prefix):
    x = layers.Conv2D(
        fpn_dim, 3, padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(),
        name=(prefix+"_conv"))(inputs)
    x = GroupNormalization(groups=num_group, name=(prefix+"_gn"))(x)
    return x

def fpn_block(inputs,
            fpn_dim=256,
            num_extra=2,
            num_group=32,
            prefix="fpn"):
    num_pyramid = len(inputs)
    outputs = []

    for lvl in range(0, num_pyramid):
        # adaptive
        name = "%s_adaptive%d" % (prefix, lvl)
        adaptive =  build_latlayer(inputs[num_pyramid-1-lvl], fpn_dim, num_group, name)

        if lvl > 0:
            name = "%s_up_%d" % (prefix, lvl)
            x = layers.UpSampling2D(2, interpolation="nearest", name=name)(x)
            
            name = "%s_fuse_%d" % (prefix, lvl)
            x = layers.Add(name=name)([x, adaptive])
        else:
            x = adaptive

        name = "%s_smooth%d" % (prefix, lvl)
        outputs.append(build_smooth_layer(x, fpn_dim, num_group, name))
    outputs = list(reversed(outputs))
    
    # build extra layer
    for i in range(num_extra):
        name = "%s_extra%d" % (prefix, i)
        if i == 0:
            x = layers.Conv2D(fpn_dim, 3, strides=2, padding=[[0,0],[1,1],[1,1],[0,0]],
                name=(name+"_conv"))(outputs[-1])
            x = GroupNormalization(groups=num_group, name=(name+"_gn"))(x)
        else:
            x = layers.ReLU(name=(name+"_relu"))(x)
            x = layers.Conv2D(fpn_dim, 3, strides=2, padding=[[0,0],[1,1],[1,1],[0,0]],
                name=(name+"_conv"))(outputs[-1])
            x = GroupNormalization(groups=num_group, name=(name+"_gn"))(x)
        outputs.append(x)

    return outputs
