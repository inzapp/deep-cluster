"""
Authors : inzapp

Github url : https://github.com/inzapp/deep-cluster

Copyright (c) 2022 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import numpy as np
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.ae_e = None
        self.ae = None

    def build(self, ae_e=None):
        if ae_e is None:
            ae_e_input, ae_e_output = self.build_ae_e()
            self.ae_e = tf.keras.models.Model(ae_e_input, ae_e_output)
        else:
            ae_e_input, ae_e_output = ae_e.input, ae_e.output
            self.ae_e = ae_e

        ae_d_input, ae_d_output = self.build_ae_d()
        ae_d = tf.keras.models.Model(ae_d_input, ae_d_output)

        ae_output = ae_d(ae_e_output)
        self.ae = tf.keras.models.Model(ae_e_input, ae_output)
        return self.ae, self.ae_e

    def build_ae_e(self):
        ae_e_input = tf.keras.layers.Input(shape=self.input_shape)
        x = ae_e_input
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            units=self.latent_dim,
            kernel_initializer='he_normal',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        ae_e_output = x
        return ae_e_input, ae_e_output

    def build_ae_d(self):
        ae_d_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = ae_d_input
        x = tf.keras.layers.Dense(
            units=int(np.prod(self.input_shape)),
            kernel_initializer='glorot_normal',
            activation='sigmoid')(x)
        x = tf.keras.layers.Reshape(target_shape=self.input_shape)(x)
        ae_d_output = x
        return ae_d_input, ae_d_output

