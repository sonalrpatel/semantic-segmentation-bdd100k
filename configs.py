# ================================================================
#   File name   : configs.py
#   Author      : sonalrpatel
#   Created date: 18-07-2022
#   GitHub      : https://github.com/sonalrpatel/semantic_segmentation
#   Description : configuration file
# ================================================================

# ================================================================
#   Possible encoder and weights combinations
# ----------------------------------------------------------------
#   Encoder         Weights
# ----------------------------------------------------------------
#   default         None        --  only for vanilla_unet
#   resnet34/50     None        --  resnet34/50 encoder (self configured) w/o any pretrained weights
#   resnet34cm      None        --  resnet34 encoder from classification_models w/o any pretrained weights
#   resnet50ka      None        --  resnet50 encoder from keras_applications w/o any pretrained weights
#   resnet34cm      imagenet    --  resnet34 encoder from classification_models with pretrained weights on imagenet
#   resnet50ka      imagenet    --  resnet50 encoder from keras_applications with pretrained weights on imagenet
# ================================================================
MODEL_NAME = 'unet'  # unet, pspnet, deeplabv3, fpn
MODEL_ENCODER = 'default'
MODEL_ENCODER_WEIGHTS = 'None'
MODEL_OPTIMIZER = 'Adam'
MODEL_LOSS = 'default'  # ce_iou

# IMAGE size
IMAGE_SIZE = (192, 192, 3)

# Dataset
DATASET = 'bdd100k'
DIR_DATA = 'data/bdd100k/'
DIR_TRAIN_IMG = DIR_DATA + 'images/train/'
DIR_TRAIN_SEG = DIR_DATA + 'colormaps/train/'
DIR_VAL_IMG = DIR_DATA + 'images/val/'
DIR_VAL_SEG = DIR_DATA + 'colormaps/val/'
PATH_CLASSES = DIR_DATA + 'class_dict.xlsx'
PATH_WEIGHTS = None
VERIFY_DATASET = False

# LOG directory
LOG_DIR = 'logs/'
LOG_DIR2 = 'logs/'  # server link to store the checkpoint
TRIAL_NO = 't1'

# TRAIN options
TRAIN_DATA_AUG = True
TRAIN_FREEZE_BODY = True
TRAIN_FREEZE_BATCH_SIZE = 32
TRAIN_UNFREEZE_BATCH_SIZE = 16  # note that more GPU memory is required after unfreezing the body
TRAIN_FREEZE_LR = 1e-3
TRAIN_UNFREEZE_LR = 1e-4
TRAIN_FREEZE_INIT_EPOCH = 0
TRAIN_FREEZE_END_EPOCH = 30
TRAIN_UNFREEZE_END_EPOCH = 50  # note that it is considered when TRAIN_FREEZE_BODY is True
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according to validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = False  # saves all best validated checkpoints in training process (False recommended)
TRAIN_FROM_CHECKPOINT = False
TRAIN_TRANSFER = True

# TEST options
TEST_DATA_AUG = False
TEST_BATCH_SIZE = 16
TEST_MODE = 'mean_iou'  # mean_iou, predict

# VAL options
VAL_DATA_AUG = False
VAL_BATCH_SIZE = 16
VAL_VALIDATION_USING = 'TRAIN'  # note that when validation data does not exist, set it to TRAIN or None
VAL_VALIDATION_SPLIT = 0.2  # note that it will be used when VAL_VALIDATION_USING is TRAIN

# Augmentation options
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, GridDropout, ColorJitter,
    RandomBrightnessContrast, RandomGamma, OneOf, Rotate, RandomSunFlare, Cutout,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, HueSaturationValue,
    RGBShift, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop
)

AUGMENTATIONS_TRAIN_SOFT = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(rotate_limit=20, shift_limit=0.07, scale_limit=0.2, p=0.3),
    OneOf([
        RandomSizedCrop(min_max_height=(96, 160), height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        Cutout(num_holes=4)
    ], p=0.2)
], p=1)

AUGMENTATIONS_TRAIN_HARD = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(rotate_limit=40, shift_limit=0.1, scale_limit=0.4, p=0.5),
    OneOf([
        RandomSizedCrop(min_max_height=(96, 144), height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        Cutout(num_holes=12),
    ], p=0.5),
    MotionBlur(p=0.4),
    OneOf([
        GridDropout(),
        ElasticTransform(),
        GridDistortion(),
        OpticalDistortion(distort_limit=1, shift_limit=0.5)
    ], p=0.2)
], p=1)

AUGMENTATION_SCHEDULE = False
AUGMENTATION_MODE = None  # AUGMENTATIONS_TRAIN_SOFT, AUGMENTATIONS_TRAIN_HARD
