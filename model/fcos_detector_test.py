import tensorflow as tf

from model.fcos_detector import build_fcos_detector

def test_fcos_detector():
    model = build_fcos_detector(80)
    model.summary()

    inputs = tf.random.normal((1, 512, 512, 3))
    outputs = model(inputs)
    print(outputs)

if __name__ == '__main__':
    test_fcos_detector()