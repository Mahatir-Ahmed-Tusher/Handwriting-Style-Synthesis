import tensorflow as tf
from tensorflow.keras import layers

class StyleEncoder(tf.keras.Model):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')
        self.res_block1 = self._residual_block(64)
        self.res_block2 = self._residual_block(64)
        self.conv3 = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same')
        self.res_block3 = self._residual_block(128)
        self.res_block4 = self._residual_block(128)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(128)  # Output style vector of size 128

    def _residual_block(self, filters):
        def block(x):
            residual = x
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)
            return x
        return block

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.conv3(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.flatten(x)
        style_vector = self.dense(x)
        return style_vector