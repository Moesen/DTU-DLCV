from keras import layers
from tensorflow import keras
import tensorflow as tf

def conv_block(
    layer_input: tf.Tensor,
    n_channels: int,
    do_batchnorm: bool,
    do_dropout: bool,
    dropout_percentage: float = .2, 
    kernel_size: int = 3, 
):
    d = layers.Conv2D(
        n_channels,
        kernel_size=kernel_size,
        strides=1,
        activation="relu",
        padding="same",
    )(layer_input)

    if do_batchnorm:
        d = layers.BatchNormalization()(d)

    if do_dropout:
        d = layers.Dropout(dropout_percentage)(d)

    d = layers.Conv2D(
        n_channels,
        kernel_size=kernel_size,
        strides=1,
        activation="relu",
        padding="same",
    )(d)

    if do_dropout:
        d = layers.Dropout(dropout_percentage)(d)

    d = layers.MaxPooling2D((2, 2))(d)
    return d

def build_model(
    first_layer_channels: int,
    num_conv_blocks: int,
    num_classes: int,
    img_shape: tuple = (32, 32, 3),
    do_batchnorm=True,
    dropout_percentage: float = .2,
    do_dropout=True,
):
    """build_model.

    :param first_layer_channels:
    :type first_layer_channels: int
    :param num_conv_blocks:
    :type num_conv_blocks: int
    :param num_classes:
    :type num_classes: int
    :param img_shape:
    :type img_shape: tuple
    :param do_batchnorm:
    :param do_dropout:
    """
    d0 = layers.Input(shape=img_shape)


    d1 = conv_block(
        d0,
        first_layer_channels,
        do_batchnorm=do_batchnorm,
        do_dropout=do_dropout,
        dropout_percentage=dropout_percentage,
        kernel_size=7,
    )

    for i in range(num_conv_blocks):
        d1 = conv_block(
            d1,
            first_layer_channels * 2 ** (i + 1),
            do_batchnorm=do_batchnorm,
            do_dropout=do_dropout,
        )

    d4 = layers.Flatten()(d1)
    d5 = layers.Dense(100, activation="relu")(d4)
    d6 = layers.Dense(num_classes)(d5)

    return keras.models.Model(inputs=d0, outputs=d6)


class ConvNet:
    def __init__(
        self,
        first_layer_channels: int,
        num_conv_blocks: int,
        num_classes: int,
        img_shape: tuple = (32, 32, 3),
        do_batchnorm=True,
        do_dropout=True,
    ) -> None:
        """
        :type first_layer_channels: int
        :type num_conv_blocks: int
        :type num_classes: int
        :type img_shape: tuple
        :param do_dropout:
        :rtype: None
        """

        self.img_shape = img_shape
        self.first_layer_channels = first_layer_channels
        self.num_conv_blocks = num_conv_blocks
        self.do_batchnorm = do_batchnorm
        self.do_dropout = do_dropout
        self.num_classes = num_classes

    def build_model(self):
        """Returns a model based on parameters given in conv net"""

        def conv_block(
            layer_input, n_channels, kernel_size=3, BN=self.do_batchnorm, DO=True
        ):
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

        d1 = conv_block(
            d0,
            self.first_layer_channels,
            kernel_size=7,
            BN=self.do_batchnorm,
            DO=self.do_dropout,
        )

        for i in range(self.num_conv_blocks):
            d1 = conv_block(
                d1,
                self.first_layer_channels * 2 ** (i + 1),
                BN=self.do_batchnorm,
                DO=self.do_dropout,
            )

        d4 = layers.Flatten()(d1)
        d5 = layers.Dense(100, activation="relu")(d4)
        d6 = layers.Dense(self.num_classes)(d5)

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
