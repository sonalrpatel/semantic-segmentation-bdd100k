import tensorflow as tf
import keras.backend as K
import numpy as np
from math import ceil
from sys import exit
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.image import ResizeMethod

from tensorflow.keras.applications import ResNet50
from classification_models.keras import Classifiers
from model.resnet50 import resnet50_encoder
from model.resnet34 import resnet34_encoder

from model.model_utils import *


# References
# Depthwise Separable Convolution https://www.youtube.com/watch?v=T7o3xvJLuHk
# Depthwise Separable Convolution https://www.youtube.com/watch?v=vCJ4magCPts


def AtrousSpatialBlock(inputs, kernel_size=(3, 3), d_rate=1, n_filters=256,
                       batch_norm=True, activation='relu'):
    conv_block_name = 'aspp_level{}_conv'.format(d_rate)
    bn_name = 'aspp_level{}_bn'.format(d_rate)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size, dilation_rate=d_rate, padding='same',
               kernel_initializer=KERNEL_INIT, name=conv_block_name)(inputs)
    if batch_norm:
        x = BatchNormalization(name=bn_name)(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x


def AtrousSpatialPyramidPooling(res, conv_filters=None):
    kernel_sizes = [1, 3, 3, 3]
    dilation_rates = [1, 3, 6, 12]
    if conv_filters is None:
        conv_filters = int(res.shape[-1] / 2)  # (e.g 256/2 = 128)

    # build atrous spatial pyramid
    b0 = AtrousSpatialBlock(res, kernel_sizes[0], dilation_rates[0], conv_filters)
    b1 = AtrousSpatialBlock(res, kernel_sizes[1], dilation_rates[1], conv_filters)
    b2 = AtrousSpatialBlock(res, kernel_sizes[2], dilation_rates[2], conv_filters)
    b3 = AtrousSpatialBlock(res, kernel_sizes[3], dilation_rates[3], conv_filters)

    # global average pooling
    b4 = AveragePooling2D(pool_size=(6, 6))(res)
    b4 = Conv2D(conv_filters, (1, 1), padding='same', use_bias=False, kernel_initializer=KERNEL_INIT,
                name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_bn', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = tf.image.resize(b4, res.shape[1:3], method=ResizeMethod.BILINEAR, name='image_pooling_upsampling')

    # aggregate atrous spatial pyramids
    aspp = Concatenate(name='aspp_concat')([b4, b0, b1, b2, b3])

    return aspp


# Deeplabv3 Decoder
def _build_deeplabv3(n_classes, image_size=(224, 224, 3), encoder=None, weights=None, name=None):
    inputs = input_tensor(image_size)
    if encoder is "resnet34":
        # f0-inputs - 192 x 192 x 3     -   1/1
        # f1-skip   -  96 x  96 x 64    -   1/2
        # f2-skip   -  48 x  48 x 64    -   1/4
        # f3-skip   -  24 x  24 x 128   -   1/8
        # f4-skip   -  12 x  12 x 512   -   1/16
        # f5-bridge -   6 x   6 x 1024  -   1/32
        [f1, f2, f3, f4, f5] = resnet34_encoder(input_tensor=inputs)
    elif encoder is "resnet50":
        # f0-inputs - 192 x 192 x 3     -   1/1
        # f1-skip   -  96 x  96 x 64    -   1/2
        # f2-skip   -  48 x  48 x 256   -   1/4
        # f3-skip   -  24 x  24 x 512   -   1/8
        # f4-skip   -  12 x  12 x 1024  -   1/16
        # f5-bridge -   6 x   6 x 2048  -   1/32
        [f1, f2, f3, f4, f5] = resnet50_encoder(input_tensor=inputs)
    elif encoder is "resnet34cm":
        ResNet34, _ = Classifiers.get('resnet34')
        resnet34 = ResNet34(input_tensor=inputs, weights=weights, include_top=False)

        f0 = resnet34.get_layer("input_1").output                       # f0-inputs - 192 x 192 x 3
        f1 = resnet34.get_layer("relu0").output                         # f1-skip   - 96  x 96  x 64
        f2 = resnet34.get_layer("stage2_unit1_relu1").output            # f2-skip   - 48  x 48  x 64
        f3 = resnet34.get_layer("stage3_unit1_relu1").output            # f3-skip   - 24  x 24  x 128
        f4 = resnet34.get_layer("stage4_unit1_relu1").output            # f4-skip   - 12  x 12  x 256
        f5 = resnet34.get_layer("relu1").output                         # f5-bridge - 6   x 6   x 512
    elif encoder is "resnet50ka":
        resnet50 = ResNet50(input_tensor=inputs, weights=weights, include_top=False)

        f0 = resnet50.get_layer("input_1").output                       # f0-inputs - 192 x 192 x 3
        f1 = resnet50.get_layer("conv1_relu").output                    # f1-skip   - 96  x 96  x 64
        f2 = resnet50.get_layer("conv2_block3_out").output              # f2-skip   - 48  x 48  x 256
        f3 = resnet50.get_layer("conv3_block4_out").output              # f3-skip   - 24  x 24  x 512
        f4 = resnet50.get_layer("conv4_block6_out").output              # f4-skip   - 12  x 12  x 1024
        f5 = resnet50.get_layer("conv5_block3_out").output              # f5-bridge -  6  x 6   x 2048
    else:
        raise "Backbone is not defined"

    res = f4

    # build deeplabv3
    aspp = AtrousSpatialPyramidPooling(res, conv_filters=256)

    x = Conv2D(256, kernel_size=1, padding='same', kernel_initializer=KERNEL_INIT, name='concat_projection')(aspp)
    x = BatchNormalization(name='concat_projection_bn', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # model head
    x = Conv2D(n_classes, kernel_size=1, kernel_initializer=KERNEL_INIT)(x)
    x = tf.image.resize(x, image_size[0:2], name="final_upsampling")
    # x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    outputs = Activation('softmax')(x)

    # create the model
    model = Model(inputs, outputs, name=name)

    # return the constructed network architecture
    return model


def deeplabv3(model_cfg):
    (n_classes, image_size, encoder, weights, model_name) = model_cfg
    model = _build_deeplabv3(n_classes, image_size, encoder, weights, model_name)

    return model
