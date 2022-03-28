from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Dropout, Concatenate, Add, UpSampling2D, Conv2DTranspose, AveragePooling2D
from tensorflow.keras.layers import Lambda, concatenate, SpatialDropout2D
from tensorflow.python.keras import callbacks

# Keras/Tensorflow vs PyTorch
# https://www.youtube.com/watch?v=EvGS3VAsG4Y
# https://adamoudad.github.io/posts/keras_torch_comparison/syntax/
# https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d
# https://towardsdatascience.com/recreating-keras-functional-api-with-pytorch-cc2974f7143c
# 3 ways to build model in tensorflow
# https://towardsdatascience.com/3-ways-to-build-neural-networks-in-tensorflow-with-the-keras-api-80e92d3b5b7e
# https://towardsdatascience.com/3-ways-to-create-a-machine-learning-model-with-keras-and-tensorflow-2-0-de09323af4d3


# KERNEL_INIT = 'he_normal'
KERNEL_INIT = 'glorot_normal'


# function that generates input tensor based on image size
def input_tensor(image_size):
    x = Input(image_size)
    return x


# function that defines one convolutional layer with certain number of filters (n_filters)
# n_filters: represents the number of output feature channels needed
def single_conv(inputs, n_filters, kernel_size=(3, 3), d_rate=1, batch_norm=False, activation=None,
                padding='same', name=None):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, dilation_rate=d_rate, padding=padding,
               kernel_initializer=KERNEL_INIT, name=name)(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


# function that defines two sequential 2D convolutional layers with certain number of filters
def double_conv(inputs, n_filters, kernel_size=(3, 3), d_rate=1, batch_norm=False, activation=None):
    x = single_conv(inputs, n_filters, kernel_size, d_rate, batch_norm, activation)
    x = single_conv(x, n_filters, kernel_size, d_rate, batch_norm, activation)
    return x


# encoder block
# x: represents the output of the conv_block, which goes as the skip connection for the corresponding decoder block
# p: represents the reduced feature maps passed to the next block as the input
def encoder_block(inputs, n_filters, kernel_size=(3, 3), pool_size=(2, 2), d_rate=1, batch_norm=True, activation='relu',
                  dropout_rate=None):
    x = double_conv(inputs, n_filters, kernel_size, d_rate, batch_norm, activation)
    p = MaxPooling2D(pool_size)(x)
    if dropout_rate is not None:
        p = Dropout(rate=dropout_rate)(p)
    return x, p


# function that merges two layers (Concatenate)
def merge(input1, input2):
    x = Concatenate()([input1, input2])
    return x


# function that defines 2D transposed convolutional (Deconvolution) layer
def de_conv(inputs, n_filters, kernel_size=(2, 2), strides=(2, 2), batch_norm=False, activation=None,
            padding='same', name=None):
    x = Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding,
                        kernel_initializer=KERNEL_INIT, name=name)(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


# decoder block
def decoder_block(inputs, skip_features, n_filters, kernel_size=(3, 3), strides=(2, 2), d_rate=1, batch_norm=True,
                  transpose_kernel_size=(2, 2), activation='relu', dropout_rate=None, block_type='transpose'):
    if block_type == 'transpose':
        x = de_conv(inputs, n_filters, transpose_kernel_size, strides, batch_norm, activation)
    else:
        x = UpSampling2D(size=strides, interpolation='bilinear')(inputs)
    x = merge(x, skip_features)
    if dropout_rate is not None:
        x = Dropout(rate=dropout_rate)(x)
    x = double_conv(x, n_filters, kernel_size, d_rate, batch_norm, activation)
    return x
