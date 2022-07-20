from tensorflow.keras.models import Model

from tensorflow.keras.applications import ResNet50
from classification_models.keras import Classifiers
from model.resnet34 import resnet34_encoder
from model.resnet50 import resnet50_encoder

from model.model_utils import *


# UNet Encoder Decoder
def _build_vanilla_unet(n_classes, input_shape=(192, 192, 3), filters=64, name=None):
    assert input_shape[0] % 16 == 0
    assert input_shape[1] % 16 == 0

    inputs = input_tensor(image_size=input_shape)

    # Contraction path
    s1, p1 = encoder_block(inputs, filters, batch_norm=True, dropout_rate=None)
    s2, p2 = encoder_block(p1, filters * 2, batch_norm=True, dropout_rate=None)
    s3, p3 = encoder_block(p2, filters * 4, batch_norm=True, dropout_rate=None)
    s4, p4 = encoder_block(p3, filters * 8, batch_norm=True, dropout_rate=None)

    c1 = double_conv(p4, filters * 16)

    d1 = decoder_block(c1, s4, filters * 8, batch_norm=True, dropout_rate=None)
    d2 = decoder_block(d1, s3, filters * 4, batch_norm=True, dropout_rate=None)
    d3 = decoder_block(d2, s2, filters * 2, batch_norm=True, dropout_rate=None)
    d4 = decoder_block(d3, s1, filters, batch_norm=True, dropout_rate=None)

    outputs = single_conv(d4, n_classes, 1, activation="softmax")

    # create the model
    model = Model(inputs, outputs, name=name)

    # return the constructed network architecture
    return model


# UNet Decoder
def _build_unet(n_classes, image_size=(192, 192, 3), encoder=None, weights=None, name=None):
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

    skip = [f1, f2, f3, f4]
    bridge = f5
    #                                                                           ResNet50    ResNet34
    d1 = decoder_block(bridge, skip[3], n_filters=skip[3].shape[-1])    # filters   1024    256
    d2 = decoder_block(d1, skip[2], n_filters=skip[2].shape[-1])        # filters   512     128
    d3 = decoder_block(d2, skip[1], n_filters=skip[1].shape[-1])        # filters   256     64
    d4 = decoder_block(d3, skip[0], n_filters=skip[0].shape[-1])        # filters   64      64

    x = Conv2DTranspose(filters=n_classes,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        kernel_initializer=KERNEL_INIT,
                        padding='same')(d4)                             # returns 192 x 192 x 38
    # x = UpSampling2D(size=(2, 2), interpolation='bilinear',
    #                  name=None)(d4)                                   # returns 192 x 192 x 64

    outputs = single_conv(x, n_classes, 1, activation="softmax")        # outputs 192 x 192 x 38

    # create the model
    model = Model(inputs, outputs, name=name)

    # return the constructed network architecture
    return model


def unet(model_cfg):
    (n_classes, image_size, encoder, weights, name) = model_cfg
    if encoder is "default":
        model = _build_vanilla_unet(n_classes, input_shape=image_size, name=name)
    else:
        model = _build_unet(n_classes, image_size=image_size, encoder=encoder, weights=weights, name=name)

    return model
