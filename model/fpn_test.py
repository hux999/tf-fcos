import tensorflow as tf

from model.fpn import fpn_block

def test_fpn():
    fpn_dim = 256
    num_pyramid = 3
    inputs = []
    for i in range(num_pyramid):
        shape = (8*(2**i), 8*(2**i), fpn_dim)
        inputs.append(tf.keras.Input(shape=shape, dtype=tf.float32))
    inputs = list(reversed(inputs))
    outputs = fpn_block(inputs, fpn_dim)

    model = tf.keras.Model(inputs, outputs)
    model.summary()


if __name__ == '__main__':
    test_fpn()