from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from classification_models.keras import Classifiers
from tensorflow.keras.applications import ResNet50

from model.resnet34 import resnet34_encoder
from model.resnet50 import resnet50_encoder

from model.model_utils import *
from model.pspnet import SpatialPyramidPooling
from model.deeplabv3 import AtrousSpatialPyramidPooling
from model.fpn import FPNBlock, DoubleConv3x3BnReLU


def fpnet(inputs, pyramid_filters=256, segmentation_filters=128,
          aggregation='concat', dropout=0.2):
    # f0-inputs - 192 x 192 x 3     -   1/1
    # f1-skip   -  96 x  96 x 64    -   1/2
    # f2-skip   -  48 x  48 x 64    -   1/4
    # f3-skip   -  24 x  24 x 128   -   1/8
    # f4-skip   -  12 x  12 x 256   -   1/16
    # f5-bridge -   6 x   6 x 512   -   1/32
    [f1, f2, f3, f4] = inputs

    # build FPN pyramid
    p4 = FPNBlock(pyramid_filters, stage=4)(skip=f4)
    p3 = FPNBlock(pyramid_filters, stage=3)(input_tensor=p4, skip=f3)
    p2 = FPNBlock(pyramid_filters, stage=2)(input_tensor=p3, skip=f2)
    p1 = FPNBlock(pyramid_filters, stage=1)(input_tensor=p2, skip=f1)

    # add segmentation head to each
    s4 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage2')(p2)
    s1 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm=True, name='segm_stage1')(p1)

    # upsampling to same resolution
    s4 = UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s4)
    s3 = UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s3)
    s2 = UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s2)

    # aggregating results
    if aggregation == 'sum':
        x = Add(name='aggregation_sum')([s1, s2, s3, s4])
    elif aggregation == 'concat':
        x = Concatenate(name='aggregation_concat')([s1, s2, s3, s4])
    else:
        raise ValueError('Aggregation parameter should be in ("sum", "concat"), '
                         'got {}'.format(aggregation))

    if dropout:
        x = SpatialDropout2D(dropout, name='pyramid_dropout')(x)

    # restore input shape
    x = Conv2D(inputs[0].shape[-1], kernel_size=1, padding='same',
               kernel_initializer=KERNEL_INIT, name='concat_covn')(x)
    x = BatchNormalization(name='concat_bn', epsilon=1e-5)(x)
    x = Activation('relu', name='concat_relu')(x)
    outputs = Dropout(0.1)(x)

    return outputs


def SpatialPyramidPooling_PostUpdate(inputs):
    psp = SpatialPyramidPooling(inputs)

    # restore input shape
    x = Conv2D(inputs.shape[-1], kernel_size=1, padding='same',
               kernel_initializer=KERNEL_INIT, name='post_spp_conv')(psp)
    x = BatchNormalization(name='post_spp_bn', epsilon=1e-5)(x)
    x = Activation('relu', name='post_spp_relu')(x)
    outputs = Dropout(0.1)(x)

    return outputs


def AtrousSpatialPyramidPooling_PostUpdate(inputs):
    aspp = AtrousSpatialPyramidPooling(inputs)

    # restore input shape
    x = Conv2D(inputs.shape[-1], kernel_size=1, padding='same',
               kernel_initializer=KERNEL_INIT, name='post_aspp_conv')(aspp)
    x = BatchNormalization(name='post_aspp_bn', epsilon=1e-5)(x)
    x = Activation('relu', name='post_aspp_relu')(x)
    outputs = Dropout(0.1)(x)

    return outputs


def AtrousSpatialPyramidPooling_SkipAdd(inputs, skip):
    aspp = AtrousSpatialPyramidPooling(inputs, conv_filters=256)

    x = Conv2D(256, kernel_size=1, padding='same', kernel_initializer=KERNEL_INIT, name='concat_projection')(aspp)
    x = BatchNormalization(name='concat_projection_bn', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # aspp head
    x = Conv2D(skip.shape[-1], kernel_size=1, kernel_initializer=KERNEL_INIT)(x)
    x = tf.image.resize(x, skip.shape[1:3], name="aspp_upsampling")
    x = Activation('relu')(x)

    outputs = Add(name='add_aspp_skip')([x, skip])

    return outputs


def AtrousSpatialPyramidPooling_PrePostUpdate(inputs, kernel_sizes, dilation_rates):
    # f0-inputs - 192 x 192 x 3     -   1/1
    # f1-skip   -  96 x  96 x 64    -   1/2
    # f2-skip   -  48 x  48 x 64    -   1/4
    # f3-skip   -  24 x  24 x 128   -   1/8
    # f4-skip   -  12 x  12 x 256   -   1/16
    # f5-bridge -   6 x   6 x 512   -   1/32
    [f1, f2, f3, f4] = inputs

    u0 = f1
    u1 = UpSampling2D((2, 2), interpolation='bilinear', name='pre_aspp_upsampling2x')(f2)
    u2 = UpSampling2D((4, 4), interpolation='bilinear', name='pre_aspp_upsampling4x')(f3)
    u3 = UpSampling2D((8, 8), interpolation='bilinear', name='pre_aspp_upsampling8x')(f4)

    b0 = single_conv(u0, 128, kernel_sizes[0], dilation_rates[0], name='pre_aspp_conv1')
    b1 = single_conv(u1, 128, kernel_sizes[1], dilation_rates[1], name='pre_aspp_conv2')
    b2 = single_conv(u2, 128, kernel_sizes[2], dilation_rates[2], name='pre_aspp_conv3')
    b3 = single_conv(u3, 128, kernel_sizes[3], dilation_rates[3], name='pre_aspp_conv4')

    c = Concatenate(name='pre_aspp_concatenation')([b0, b1, b2, b3])
    c = SpatialDropout2D(0.2)(c)

    aspp = AtrousSpatialPyramidPooling(c)

    # restore input shape
    x = Conv2D(inputs[0].shape[-1], kernel_size=1, padding='same',
               kernel_initializer=KERNEL_INIT, name='post_aspp_conv')(aspp)
    x = BatchNormalization(name='post_aspp_bn', epsilon=1e-5)(x)
    x = Activation('relu', name='post_aspp_relu')(x)
    outputs = Dropout(0.1)(x)

    return outputs


def fpn_aspp(inputs):
    # f0-inputs - 192 x 192 x 3     -   1/1
    # f1-skip   -  96 x  96 x 64    -   1/2
    # f2-skip   -  48 x  48 x 64    -   1/4
    # f3-skip   -  24 x  24 x 128   -   1/8
    # f4-skip   -  12 x  12 x 256   -   1/16
    # f5-bridge -   6 x   6 x 512   -   1/32
    [f1, f2, f3, f4, f5] = inputs

    # build FPN pyramid
    p4 = FPNBlock(f4.shape[-1], stage=4)(input_tensor=f5, skip=f4)
    p3 = FPNBlock(f3.shape[-1], stage=3)(input_tensor=p4, skip=f3)
    p2 = FPNBlock(f2.shape[-1], stage=2)(input_tensor=p3, skip=f2)
    p1 = FPNBlock(f1.shape[-1], stage=1)(input_tensor=p2, skip=f1)

    aspp = AtrousSpatialPyramidPooling_PostUpdate(p1)

    return [aspp, p2, p3, p4]


# UNet Decoder
def _build_unet_adv(n_classes, image_size=(192, 192, 3), encoder=None, weights=None, name=None):
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

    if name == "resnet34pt_unet_adv_1":
        skip = [f1, f2, f3, f4]
        bridge = SpatialPyramidPooling(f5)                              # 6   x 6   x 1024
    elif name == "resnet34pt_unet_adv_2":
        skip = [f1, f2, f3, f4]
        bridge = AtrousSpatialPyramidPooling(f5)                        # 6   x 6   x 1280
    elif name == "resnet34pt_unet_adv_3":
        f2 = SpatialPyramidPooling_PostUpdate(f2)                       # 48  x 48  x 64
        skip = [f1, f2, f3, f4]
        bridge = f5
    elif name == "resnet34pt_unet_adv_4":
        f1 = AtrousSpatialPyramidPooling_SkipAdd(f4, f1)                # 96  x 96  x 64
        skip = [f1, f2, f3, f4]
        bridge = f5
    elif name == "resnet34pt_unet_adv_5":
        skip = [f1, f2, f3, f4]
        f1 = fpnet(skip)                                                  # 96  x 96  x 64
        skip = [f1, f2, f3, f4]
        bridge = f5
    elif name == "resnet34pt_unet_adv_6":
        skip = [f1, f2, f3, f4]
        kernel_sizes = [1, 3, 3, 3]
        dilation_rates = [1, 3, 6, 12]                                  # 96  x 96  x 64
        f1 = AtrousSpatialPyramidPooling_PrePostUpdate(skip, kernel_sizes, dilation_rates)
        skip = [f1, f2, f3, f4]
        bridge = f5
    elif name == "resnet34pt_unet_adv_7":
        skip = [f1, f2, f3, f4]
        kernel_sizes = [3, 3, 3, 1]
        dilation_rates = [12, 6, 3, 1]                                  # 96  x 96  x 64
        f1 = AtrousSpatialPyramidPooling_PrePostUpdate(skip, kernel_sizes, dilation_rates)
        skip = [f1, f2, f3, f4]
        bridge = f5
    elif name == "resnet34pt_unet_adv_8":
        skip = [f1, f2, f3, f4, f5]
        skip = fpn_aspp(skip)                                           # skip [f1, f2, f3, f4]
        bridge = f5
    else:
        exit(0)

    d1 = decoder_block(bridge, skip[3], n_filters=skip[3].shape[-1])    # 12  x 12  x 256
    d2 = decoder_block(d1, skip[2], n_filters=skip[2].shape[-1])        # 24  x 24  x 128
    d3 = decoder_block(d2, skip[1], n_filters=skip[1].shape[-1])        # 48  x 48  x 64
    d4 = decoder_block(d3, skip[0], n_filters=skip[0].shape[-1])        # 96  x 96  x 64

    x = Conv2DTranspose(n_classes, kernel_size=3, strides=2,
                        kernel_initializer=KERNEL_INIT, padding='same')(d4)     # returns 192 x 192 x 38
    x = Conv2D(n_classes, kernel_size=1, kernel_initializer=KERNEL_INIT)(x)     # returns 192 x 192 x 38
    outputs = Activation('softmax')(x)

    # x = Conv2D(n_classes, kernel_size=1, kernel_initializer=KERNEL_INIT)(d4)    # returns 96  x 96  x 38
    # x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)                  # returns 192 x 192 x 38
    # outputs = Activation('softmax')(x)

    # create the model
    model = Model(inputs, outputs, name=name)

    # return the constructed network architecture
    return model


def unet_adv(model_cfg):
    (n_classes, image_size, encoder, weights, model_name) = model_cfg
    model = _build_unet_adv(n_classes, image_size, encoder, weights, model_name)

    return model
