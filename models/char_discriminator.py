class CharacterDiscriminator(tf.keras.Model):
    def __init__(self):
        super(CharacterDiscriminator, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')  # Binary classifier

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        output = self.dense(x)
        return output