#!/usr/bin/env python3.7

"""Layers."""

# -- Third party modules --
import numpy as np

#- Proprietary modules --
from nnumpyactivations import _ACTIVATIONS

# -- File info --
__version__ = '0.1.0'
__author__ = 'Andrzej Kucik'
__date__ = '2019-10-01'


class Layer:
    def __init__(self, trainable=False, name='', **kwargs):
        self.trainable = trainable
        self.name = name

        self.input = None
        self.input_shape = None

        self.output = None
        self.output_shape = None

        self.weights = []

        self._value = None

    def build(self):
        pass

    def call(self, inputs):
        self._value = inputs
        return inputs

    def compute_output_shape(self, input_shape):

        return input_shape

    def get_config(self):
        return {'trainable': self.trainable,
                'name': self.name,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape}

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        if not isinstance(weights, list):
            exit('Weights must be of instance `list`!')

        if len(weights) != self.weights:
            exit('Weights lists` lengths do not match!')

        for n in range(len(weights)):
            if weights[n].shape != self.weights[n].shape:
                exit('Weights` shapes do not match!')


class Input(Layer):
    def __init__(self, input_shape, **kwargs):
        super(Input, self).__init__(**kwargs)
        self.trainable = False
        self.input_shape = input_shape
        self.output_shape = input_shape


class Dense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self._units = units
        if activation is None:
            activation = ''
        self._activation = _ACTIVATIONS[activation.lower()]()

    def build(self):
        limit = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.output_shape)))
        w = np.random.uniform(-limit, limit, size=self.input_shape + (self._units,)).astype('float32')
        b = np.zeros(self.output_shape, dtype='float32')

        self.weights = [w, b]

    def call(self, inputs):
        self._value = self._activation.call(np.dot(inputs, self.weights[0]) + self.weights[1])

        return self._value

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self._units,)

    def get_config(self):
        return {'trainable': self.trainable,
                'name': self.name,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'units': self._units,
                'activation': self._activation.name}
