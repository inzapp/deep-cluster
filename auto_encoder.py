"""
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
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class AutoEncoder:
    def __init__(self, input_shape, encoding_dim):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.latent_rows = self.input_shape[0] // 8
        self.latent_cols = self.input_shape[1] // 8
        self.latent_channels = 8192 // (self.latent_rows * self.latent_cols)
        if self.latent_channels > 128:
            self.latent_channels = 128
        encoder_input, encoder_output, ae_output = self.__convolutional_ae()
        self.encoder = tf.keras.models.Model(encoder_input, encoder_output)
        self.ae = tf.keras.models.Model(encoder_input, ae_output)
        self.ae.save('ae.h5', include_optimizer=False)

    def __convolutional_ae(self):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = encoder_input
        x = tf.keras.layers.Conv2D(strides=2, filters=32,  kernel_size=3, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(strides=2, filters=64,  kernel_size=3, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(strides=2, filters=128, kernel_size=3, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(strides=1, filters=self.latent_channels, kernel_size=1, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=2048, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.Dense(units=512, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.Dense(units=self.encoding_dim, kernel_initializer='glorot_normal', activation='linear')(x)
        encoder_output = x

        x = tf.keras.layers.Dense(units=512, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.Dense(units=2048, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.Dense(units=self.latent_rows * self.latent_cols * self.latent_channels, activation='relu')(x)
        x = tf.keras.layers.Reshape(target_shape=(self.latent_rows, self.latent_cols, self.latent_channels))(x)
        x = tf.keras.layers.Conv2DTranspose(strides=1, filters=128, kernel_size=1, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(strides=2, filters=128, kernel_size=3, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(strides=2, filters=64,  kernel_size=3, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(strides=2, filters=32,  kernel_size=3, kernel_initializer='he_normal', padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(strides=1, filters=self.input_shape[-1], kernel_size=1, kernel_initializer='glorot_normal', padding='same', activation='sigmoid')(x)
        cae_output = x
        return encoder_input, encoder_output, cae_output

    def __dense_ae(self):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(encoder_input)
        x = tf.keras.layers.Dense(
            units=4096,
            kernel_initializer='he_normal',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        encoder_output = tf.keras.layers.Dense(
            units=self.encoding_dim,
            activation='linear')(x)
        x = tf.keras.layers.Dense(
            units=4096,
            kernel_initializer='he_normal',
            use_bias=False)(encoder_output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(
            units=int(self.input_shape[0] * self.input_shape[1] * self.input_shape[2]),
            activation='sigmoid')(x)
        ae_output = tf.keras.layers.Reshape(target_shape=self.input_shape)(x)
        return encoder_input, encoder_output, ae_output

