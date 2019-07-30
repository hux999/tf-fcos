import tensorflow as tf

from layers.group_normalization import GroupNormalization


def test_group_normalization_eager_mode():
    group_norm = GroupNormalization()
    x = tf.random.uniform((10, 10, 10, 256))
    group_norm(x)

def test_group_normalization_symbol_mode():
    inputs = tf.keras.Input(shape=(10, 10, 256), dtype=tf.float32)
    outputs = GroupNormalization()(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    x = tf.random.uniform((10, 10, 10, 256))
    y = model(x)
    print(y)
    

if __name__ == '__main__':
    test_group_normalization_eager_mode()
    test_group_normalization_symbol_mode()