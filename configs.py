# ================================================================
#   File name   : configs.py
#   Author      : sonalrpatel
#   Created date: 18-07-2022
#   GitHub      : https://github.com/sonalrpatel/semantic_segmentation
#   Description : configuration file
# ================================================================

# Model Options
# Update the path & others as per training trials
MODEL_NAME = "resnet34pt_unet"
MODEL_ENCODER = "default"  # default, resnet34, resnet34cm, resnet50, resnet50ka
MODEL_WEIGHTS = None  # None, imagenet
MODEL_LOSS = "default"  # "ce_iou"
MODEL_OPTIMIZER = "Adam"

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
PATH_WEIGHT = None
VERIFY_DATASET = False

# LOG directory
LOG_DIR = "logs/"
LOG_DIR2 = "logs/"  # server link to store the checkpoint
TRIAL_NO = 'trial_1'

# TRAIN options
TRAIN_AUG = True
TRAIN_AUG_SCHEDULE = False
TRAIN_AUG_MODE = 'soft'  # 'hard'
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
TEST_MODE = "mean_iou"  # mean_iou, predict
TEST_BATCH_SIZE = 16
TEST_DATA_AUG = False

# VAL options
VAL_DATA_AUG = False
VAL_BATCH_SIZE = 16
VAL_VALIDATION_USING = "TRAIN"  # note that when validation data does not exist, set it to TRAIN or None
VAL_VALIDATION_SPLIT = 0.2  # note that it will be used when VAL_VALIDATION_USING is TRAIN

# PREDICT options
