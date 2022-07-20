import numpy as np
import tensorflow as tf
import keras.backend as K
from math import ceil
from tensorflow.keras.models import Model

from tensorflow.keras.applications import ResNet50
from classification_models.keras import Classifiers
from model.resnet50 import resnet50_encoder
from model.resnet34 import resnet34_encoder

from model.model_utils import *

# references
# https://github.com/mapbox/robosat/issues/60


def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    def wrapper(input_tensor):
        return single_conv(input_tensor, filters,
                           kernel_size=(3, 3),
                           batch_norm=use_batchnorm,
                           activation='relu',
                           padding='same',
                           name=name)

    return wrapper


def DoubleConv3x3BnReLU(filters, use_batchnorm, name=None):
    name1, name2 = None, None
    if name is not None:
        name1 = name + 'a'
        name2 = name + 'b'

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name1)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name2)(x)
        return x

    return wrapper


def FPNBlock(pyramid_filters, stage):
    conv0_name = 'fpn_stage_p{}_pre_conv'.format(stage)
    conv1_name = 'fpn_stage_p{}_conv'.format(stage)
    add_name = 'fpn_stage_p{}_add'.format(stage)
    up_name = 'fpn_stage_p{}_upsampling'.format(stage)

    def wrapper(input_tensor=None, skip=None):
        if skip is not None:
            skip = single_conv(skip,
                               n_filters=pyramid_filters,
                               kernel_size=(1, 1),
                               name=conv1_name)

        if input_tensor is not None:
            input_tensor = single_conv(input_tensor,
                                       n_filters=pyramid_filters,
                                       kernel_size=(1, 1),
                                       name=conv0_name)

            x = UpSampling2D((2, 2), name=up_name)(input_tensor)
            x = Add(name=add_name)([x, skip])
        else:
            x = skip

        return x

    return wrapper


# FPN Decoder
def _build_fpn(n_classes, input_shape=(192, 192, 3), encoder=None,
               pyramid_filters=256, segmentation_filters=128,
               aggregation='concat', dropout=None,  weights=None, name=None):
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

    skips = [f5, f4, f3, f2]

    # build FPN pyramid
    p5 = FPNBlock(pyramid_filters, stage=5)(skip=skips[0])
    p4 = FPNBlock(pyramid_filters, stage=4)(input_tensor=p5, skip=skips[1])
    p3 = FPNBlock(pyramid_filters, stage=3)(input_tensor=p4, skip=skips[2])
    p2 = FPNBlock(pyramid_filters, stage=2)(input_tensor=p3, skip=skips[3])

    # add segmentation head to each
    s5 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage2')(p2)

    # upsampling to same resolution
    s5 = UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s5)
    s4 = UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s4)
    s3 = UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s3)

    # aggregating results
    if aggregation == 'sum':
        x = Add(name='aggregation_sum')([s2, s3, s4, s5])
    elif aggregation == 'concat':
        x = Concatenate(name='aggregation_concat')([s2, s3, s4, s5])
    else:
        raise ValueError('Aggregation parameter should be in ("sum", "concat"), '
                         'got {}'.format(aggregation))

    if dropout:
        x = SpatialDropout2D(dropout, name='pyramid_dropout')(x)

    # final stage
    x = Conv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='final_stage')(x)
    x = UpSampling2D((4, 4), interpolation='bilinear', name='final_upsampling')(x)

    # model head (define number of output classes)
    outputs = single_conv(x, n_classes, 3, name='head_conv', activation='softmax')

    # create the model
    model = Model(inputs, outputs, name=name)

    # return the constructed network architecture
    return model


def fpn(model_cfg):
    (n_classes, image_size, encoder, weights, name) = model_cfg
    model = _build_fpn(n_classes, input_shape=image_size, encoder=encoder, weights=weights, name=name)

    return model