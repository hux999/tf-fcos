import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec


class ScaleLayer(Layer):

    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = [1] * len(input_shape)
        self.scale = self.add_weight(shape=shape, name='scale')

    def call(self, inputs):
        outputs = self.scale * inputs
        return outputs

