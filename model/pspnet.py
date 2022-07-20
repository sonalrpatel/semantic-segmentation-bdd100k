import numpy as np
import tensorflow as tf
import keras.backend as K
from math import ceil
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from tensorflow.keras.applications import ResNet50
from classification_models.keras import Classifiers
from model.resnet50 import resnet50_encoder
from model.resnet34 import resnet34_encoder

from model.model_utils import *


# references
# https://medium.com/analytics-vidhya/semantic-segmentation-in-pspnet-with-implementation-in-keras-4843d05fc025


def SpatialPoolingBlock(inputs, pool_factor, n_filters=512, pooling_type='avg', batch_norm=True):
    if pooling_type not in ('max', 'avg'):
        raise ValueError('Unsupported pooling type - `{}`.'.format(pooling_type) + 'Use `avg` or `max`.')

    Pooling2D = MaxPooling2D if pooling_type == 'max' else AveragePooling2D

    pooling_name = 'psp_level{}_pooling'.format(pool_factor)
    conv_block_name = 'psp_level{}_conv'.format(pool_factor)
    upsampling_name = 'psp_level{}_upsampling'.format(pool_factor)

    # extract input feature maps size (h, and w dimensions)
    spatial_size = inputs.shape[1:3]

    # Compute the kernel and stride sizes according to how large the final feature map will be
    # When the kernel factor and strides are equal, then we can compute the final feature map size
    # by simply dividing the input size by the kernel or stride factor
    # After pooling, the feature map sizes shall be 1x1, 2x2, 3x3, and 6x6.
    pool_size = (ceil(spatial_size[0] / pool_factor), ceil(spatial_size[1] / pool_factor))
    x = Pooling2D(pool_size, strides=pool_size, padding='same', name=pooling_name)(inputs)

    # Convoluted with 1x1 kernel to generate 1/4th of input feature maps (e.g 2048/4 = 512)
    x = single_conv(x, n_filters, kernel_size=(1, 1), batch_norm=batch_norm, name=conv_block_name)

    # Upsampled / Resized to restore the same size as input size using bilinear interpolation
    # x = UpSampling2D(up_size, interpolation='bilinear', name=upsampling_name)(x)
    # x = Lambda(lambda x: tf.image.resize(x, spatial_size))(x)
    x = tf.image.resize(x, spatial_size, name=upsampling_name)

    return x


def SpatialPyramidPooling(res, conv_filters=None):
    pool_factors = [1, 2, 3, 6]
    if conv_filters is None:
        conv_filters = int(res.shape[-1] / 4)  # (e.g 2048/4 = 512)

    # build spatial pyramid
    x1 = SpatialPoolingBlock(res, pool_factors[0], conv_filters)
    x2 = SpatialPoolingBlock(res, pool_factors[1], conv_filters)
    x3 = SpatialPoolingBlock(res, pool_factors[2], conv_filters)
    x6 = SpatialPoolingBlock(res, pool_factors[3], conv_filters)

    # aggregate spatial pyramid
    psp = Concatenate(name='psp_concat')([res, x1, x2, x3, x6])

    return psp


# PSPNet Decoder
def _build_pspnet(n_classes, input_shape=(192, 192, 3), encoder=None, weights=None, name=None):
    inputs = input_tensor(input_shape)
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

    # build pspnet
    psp = SpatialPyramidPooling(res, conv_filters=256)

    x = Conv2D(256, kernel_size=1, padding="same", kernel_initializer=KERNEL_INIT, name="conv5_4")(psp)
    x = BatchNormalization(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # model head
    x = Conv2D(n_classes, kernel_size=1, kernel_initializer=KERNEL_INIT)(x)
    x = tf.image.resize(x, input_shape[0:2], name="final_upsampling")
    # x = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    outputs = Activation('softmax')(x)

    # create the model
    model = Model(inputs, outputs, name=name)

    # return the constructed network architecture
    return model


def pspnet(model_cfg):
    (n_classes, image_size, encoder, weights, name) = model_cfg
    model = _build_pspnet(n_classes, input_shape=image_size, encoder=encoder, weights=weights, name=name)

    return model
