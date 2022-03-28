import numpy as np
from model.model_utils import *


# Source:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# https://www.codeproject.com/Articles/1248963/Deep-Learning-using-Python-plus-Keras-Chapter-Re
# https://github.com/qubvel/classification_models/blob/a0f006e05485a34ccf871c421279864b0ccd220b/classification_models/models/resnet.py


def conv_block(input_tensor, filters, stage, block, kernel_size=None, strides=(2, 2), d_rates=None):
    """ The conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    if kernel_size is None:
        kernel_size = [1, 3, 1]
    if d_rates is None:
        d_rates = [1, 1, 1]
    if np.mean(d_rates) > 1:
        strides = (1, 1)

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters[0], kernel_size[0], dilation_rate=d_rates[0],
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size[1], dilation_rate=d_rates[1], strides=strides, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size[2], dilation_rate=d_rates[2],
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # shortcut
    s = Conv2D(filters[2], kernel_size[2], strides=strides,
               name=conv_name_base + '1')(input_tensor)
    s = BatchNormalization(name=bn_name_base + '1')(s)

    x = Add()([x, s])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, filters, stage, block, kernel_size=None, d_rates=None):
    """ The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    if kernel_size is None:
        kernel_size = [1, 3, 1]
    if d_rates is None:
        d_rates = [1, 1, 1]

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters[0], kernel_size[0], dilation_rate=d_rates[0],
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size[1], dilation_rate=d_rates[1], padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size[2], dilation_rate=d_rates[2],
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # shortcut
    s = input_tensor

    x = Add()([x, s])
    x = Activation('relu')(x)

    return x


# encoder based on resnet50
def resnet50_encoder(input_tensor, use_dilation=False):
    assert input_tensor.shape[1] % 32 == 0     # image height
    assert input_tensor.shape[2] % 32 == 0     # image width

    # stage-1
    x = ZeroPadding2D((3, 3))(input_tensor)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    f1 = x

    # stage-2
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    d_rates = [1, 1, 1]
    x = conv_block(x, [64, 64, 256], stage=2, block='a', strides=(1, 1), d_rates=d_rates)
    x = identity_block(x, [64, 64, 256], stage=2, block='b', d_rates=d_rates)
    x = identity_block(x, [64, 64, 256], stage=2, block='c', d_rates=d_rates)
    f2 = x

    # stage-3
    x = conv_block(x, [128, 128, 512], stage=3, block='a', d_rates=d_rates)
    x = identity_block(x, [128, 128, 512], stage=3, block='b', d_rates=d_rates)
    x = identity_block(x, [128, 128, 512], stage=3, block='c', d_rates=d_rates)
    x = identity_block(x, [128, 128, 512], stage=3, block='d', d_rates=d_rates)
    f3 = x

    # stage-4
    if use_dilation:
        d_rates = [1, 2, 1]
    x = conv_block(x, [256, 256, 1024], stage=4, block='a', d_rates=d_rates)
    x = identity_block(x, [256, 256, 1024], stage=4, block='b', d_rates=d_rates)
    x = identity_block(x, [256, 256, 1024], stage=4, block='c', d_rates=d_rates)
    x = identity_block(x, [256, 256, 1024], stage=4, block='d', d_rates=d_rates)
    x = identity_block(x, [256, 256, 1024], stage=4, block='e', d_rates=d_rates)
    x = identity_block(x, [256, 256, 1024], stage=4, block='f', d_rates=d_rates)
    f4 = x

    # stage-5
    if use_dilation:
        d_rates = [1, 4, 1]
    x = conv_block(x, [512, 512, 2048], stage=5, block='a', d_rates=d_rates)
    x = identity_block(x, [512, 512, 2048], stage=5, block='b', d_rates=d_rates)
    x = identity_block(x, [512, 512, 2048], stage=5, block='c', d_rates=d_rates)
    f5 = x

    # stage-6
    x = AveragePooling2D((6, 6), name='avg_pool')(x)
    f6 = x

    return [f1, f2, f3, f4, f5]
