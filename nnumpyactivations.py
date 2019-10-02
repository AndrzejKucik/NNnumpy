#!/usr/bin/env python3.7

"""Activations."""

# -- Third party modules --
import numpy as np

# -- File info --
__version__ = '0.1.0'
__author__ = 'Andrzej Kucik'
__date__ = '2019-10-01'


class Activation:
    def __init__(self):
        self.name = 'linear'

    @staticmethod
    def call(inputs):
        return inputs

    @staticmethod
    def get_gradients(outputs):
        return np.stack([np.eye(outputs.shape[-1]) for _ in range(outputs.shape[0])])


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()
        self.name = 'relu'

    def call(self, inputs):
        return np.maximum(0, inputs)

    def get_gradients(self, outputs):
        grads = outputs > 0
        return np.stack([np.diagflat(grads[m]) for m in range(outputs.shape[0])])


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()
        self.name = 'softmax'

    def call(self, inputs):
        # Stabilize by subtracting the maximum of each example
        expn = np.exp(inputs - np.expand_dims(np.max(inputs, axis=-1), axis=-1))
        return np.divide(expn, np.expand_dims(np.sum(expn, axis=-1), axis=-1))

    def get_gradients(self, outputs):
        grads = np.dot(np.expand_dims(outputs, axis=-1), np.ones((1,) + outputs.shape[1:]))
        return grads * (np.eye(outputs.shape[-1], dtype='float16') - np.swapaxes(grads, axis1=-1, axis2=-2))


_ACTIVATIONS = {'relu': ReLU, 'softmax': Softmax, '': Activation, 'linear' : Activation}
