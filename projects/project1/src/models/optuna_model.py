import tensorflow as tf
from keras import backend as K
from keras import layers, regularizers
from tensorflow import keras


def conv_block(
    layer_input: tf.Tensor,
    n_channels: int,
    do_batchnorm: bool,
    do_dropout: bool,
    dropout_percentage: float = 0.2,
    kernel_size: int = 3,
    num_kernels: int = 2,
    kernel_regularizer_strength: float = 1e-3,
    kernel_initializer: str = "he_normal"
    ):

    d = layer_input
    for _ in range(num_kernels):            
        d = layers.Conv2D(
            n_channels,
            kernel_size=kernel_size,
            strides=1,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(kernel_regularizer_strength),
            kernel_initializer=kernel_initializer
        )(layer_input)

        if do_batchnorm:
            d = layers.BatchNormalization()(d)

        if do_dropout:
            d = layers.Dropout(dropout_percentage)(d)

    d = layers.MaxPooling2D((2, 2))(d)
    return d


def build_model(
    first_layer_channels: int,
    num_conv_blocks: int,
    num_classes: int,
    img_shape: tuple, 
    do_batchnorm=True,
    dropout_percentage: float = 0.2,
    do_dropout=True,
    num_kernels: int = 2,
    kernel_regularizer_strength: float = 1e-3,
    kernel_initializer: str = "he_normal"
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
    :format (32, 32, 3), should be based on resize in dataloader
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
        kernel_regularizer_strength=kernel_regularizer_strength,
        kernel_initializer=kernel_initializer
        
    )

    for i in range(num_conv_blocks):
        d1 = conv_block(
            d1,
            first_layer_channels * 2 ** (i + 1),
            do_batchnorm=do_batchnorm,
            do_dropout=do_dropout,
	    num_kernels=num_kernels,
        )

    d4 = layers.Flatten()(d1)
    d5 = layers.Dense(100, activation="relu")(d4)
    d6 = layers.Dense(num_classes)(d5)

    return keras.models.Model(inputs=d0, outputs=d6)
