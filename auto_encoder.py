import tensorflow as tf


class AutoEncoder:
    def __init__(self, input_shape, encoding_dim):
        self.scale_factor = 4
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        encoder_input, encoder_output, ae_output = self.__dense_ae()
        self.encoder = tf.keras.models.Model(encoder_input, encoder_output)
        self.cae = tf.keras.models.Model(encoder_input, ae_output)

    def __convolutional_ae(self):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            kernel_initializer='he_uniform',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            units=self.encoding_dim,
            activation='sigmoid')(x)
        encoder_output = x

        scaled_feature_map_rows = int(self.input_shape[0] / self.scale_factor)
        scaled_feature_map_cols = int(self.input_shape[1] / self.scale_factor)
        x = tf.keras.layers.Dense(
            units=32 * scaled_feature_map_rows * scaled_feature_map_cols,
            kernel_initializer='he_uniform',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Reshape(target_shape=(scaled_feature_map_rows, scaled_feature_map_cols, 32))(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=1,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        cae_output = tf.keras.layers.Conv2D(
            filters=self.input_shape[2],
            kernel_size=1,
            activation='sigmoid')(x)
        return encoder_input, encoder_output, cae_output

    def __dense_ae(self):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(encoder_input)
        x = tf.keras.layers.Dense(
            units=4096,
            kernel_initializer='he_uniform',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        encoder_output = tf.keras.layers.Dense(
            units=self.encoding_dim,
            activation='sigmoid')(x)
        x = tf.keras.layers.Dense(
            units=4096,
            kernel_initializer='he_uniform',
            use_bias=False)(encoder_output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(
            units=int(self.input_shape[0] * self.input_shape[1] * self.input_shape[2]),
            activation='sigmoid')(x)
        ae_output = tf.keras.layers.Reshape(target_shape=self.input_shape)(x)
        return encoder_input, encoder_output, ae_output
