import tensorflow as tf

from model.fcos_head import FCOSHead, fcos_block, compute_locations

def test_fcos_head_eager_mode():
    model = FCOSHead(256, 10)
    x = tf.random.uniform((10, 10, 10, 256))
    model(x)

def test_fcos_head_symbol_mode():
    model = FCOSHead(256, 10)
    inputs = tf.keras.Input(shape=(10, 10, 256), dtype=tf.float32)
    outputs = model(inputs)
    model.summary()

def test_fcos_block():
    conv_dim = 256
    num_pyramid = 3
    inputs = []
    for i in range(num_pyramid):
        shape = (8*(2**i), 8*(2**i), conv_dim)
        inputs.append(tf.keras.Input(shape=shape, dtype=tf.float32))
    inputs = list(reversed(inputs))
    outputs = fcos_block(inputs, conv_dim, 20)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

def test_compute_locations():
    x = tf.random.uniform((1, 10, 10, 256))
    locations = compute_locations(x, 8)
    print(locations.numpy())
    print(locations.shape, tf.shape(x))

if __name__ == '__main__':
    test_fcos_head_eager_mode()
    test_fcos_head_symbol_mode()
    test_fcos_block()
    test_compute_locations()