from tensorflow import keras
from keras import layers


class ConvNet:
    def __init__(self, f, nb, nc, img_shape: tuple = (32, 32, 3), BN = True, DO = True):
        self.img_shape = img_shape
        self.n_filters = f #32
        self.n_blocks = nb #3
        self.BN = BN
        self.DO = DO
        self.n_classes = nc

    def build_model(self):
        """Text to see this is one function"""

        def conv_block(layer_input, n_channels, kernel_size=3, BN=self.BN, DO=True):
            d = layers.Conv2D(
                n_channels,
                kernel_size=kernel_size,
                strides=1,
                activation="relu",
                padding="same",
            )(layer_input)
            if BN:
                d = layers.BatchNormalization()(d)
            if DO:
                d = layers.Dropout(0.2)(d)
            d = layers.Conv2D(
                n_channels,
                kernel_size=kernel_size,
                strides=1,
                activation="relu",
                padding="same",
            )(d)
            if DO:
                d = layers.Dropout(0.2)(d)
            d = layers.MaxPooling2D((2, 2))(d)
            return d

        d0 = layers.Input(shape=self.img_shape)

        d1 = conv_block(d0, self.n_filters, kernel_size=7, BN=self.BN, DO=self.DO)

        for i in range(self.n_blocks):
            d1 = conv_block(d1, self.n_filters * 2 ** (i + 1), BN=self.BN, DO=self.DO)

        d4 = layers.Flatten()(d1)
        d5 = layers.Dense(100, activation="relu")(d4)
        d6 = layers.Dense(self.n_classes)(d5)

        return keras.models.Model(inputs=d0, outputs=d6)


# What final model might look like using standard parameters
"""model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)) )
model.add(layers.Dropout(.2) )
model.add(layers.Conv2D(32, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu') )
model.add(layers.Dropout(.2) )
model.add(layers.Conv2D(64, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu') )
model.add(layers.Dropout(.2) )
model.add(layers.Conv2D(128, (3, 3), activation='relu') )

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10))"""
