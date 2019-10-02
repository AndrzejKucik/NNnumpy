#!/usr/bin/env python3.7

"""Models.."""

# -- Third party modules --
import numpy as np

# -- Proprietary modules --
import nnumpylayers

# -- File info --
__version__ = '0.1.0'
__author__ = 'Andrzej Kucik'
__date__ = '2019-10-01'


class Sequential:
    def __init__(self, layers, name='', input_shape=None):
        self.name = name
        self.input = nnumpylayers.Input(input_shape=input_shape, name='input')
        self.input_shape = input_shape

        self.layers = [self.input]

        for layer in layers:
            layer.input = self.layers[-1]
            self.layers[-1].output = layer
            layer.input_shape = layer.input.output_shape
            layer.output_shape = layer.compute_output_shape(layer.input_shape)
            layer.build()
            self.layers.append(layer)

    def summary(self):
        print('Layer name: \t\t Input shape: \t\t Output shape', end='\n')
        for layer in self.layers:
            print('{} \t\t {} \t\t {}'.format(layer.name, layer.input_shape, layer.output_shape))

    def predict(self, tensor):
        for layer in self.layers:
            tensor = layer.call(tensor)

        return tensor
