import os
from sys import exit
from datetime import datetime
from pathlib import Path as path

from train import train
from test import test

# References
# https://github.com/divamgupta/image-segmentation-keras/blob/dc830bbd76371aaedbf8cb997bdedca388c544c4/keras_segmentation/train.py
# https://github.com/davidtvs/Keras-LinkNet/blob/538ec6fcaf88bb9365508c891f632f4cd42da886/main.py
# https://github.com/Anshul12256/Image-Segmentation-on-CamVid-with-Variants-of-UNet
# https://www.kaggle.com/mukulkr/camvid-segmentation-using-unet
# https://www.kaggle.com/meaninglesslives/nested-unet-with-efficientnet-encoder

# Update the path & others as per training trials
dataset = 'bdd100k'
primary_path = "Fill the path to dataset"
seg_ext = "_train_color"

model_name = "resnet34pt_unet"
trial_num = "trial_1"
images_path = path(primary_path + "/images/train_")
segs_path = path(primary_path + "/color_labels/train_")
val_images_path = path(primary_path + "/images/val")
val_segs_path = path(primary_path + "/color_labels/val")
test_images_path = path(primary_path + "/images/test_")
test_segs_path = path(primary_path + "/color_labels/test_")
class_path = path(primary_path + "/class_dict_20.xlsx")
checkpoint_path = primary_path + "/results/" + model_name + "_" + trial_num + "/"

image_size = (192, 192, 3)
n_classes = 20
epochs = 40
batch_size = 32
val_batch_size = 32
test_batch_size = 32
mode = "full"               # train, test, full
test_mode = "meaniou"       # meaniou, predict
validate_using = "val"
verify_dataset = False

encoder = "default"         # default, resnet34, resnet34cm, resnet50, resnet50ka
weights = None              # None, imagenet
resume_checkpoint = False
model_weights = None
loss_type = "default"       # "ceiou"
callback_fn = "all"         # all, None
optimizer_name = "Adam"
augment_schedule = False
augmentation_mode = "soft"  # "soft", "hard"

if __name__ == '__main__':
    start = datetime.now()

    if mode in ('train', 'full'):
        model = train(model_name,
                      images_path,
                      segs_path,
                      seg_ext,
                      val_images_path,
                      val_segs_path,
                      class_path,
                      image_size,
                      n_classes,
                      validate_using,
                      verify_dataset,
                      epochs,
                      batch_size,
                      val_batch_size,
                      loss_type,
                      optimizer_name,
                      callback_fn,
                      checkpoint_path,
                      encoder,
                      weights,
                      resume_checkpoint,
                      model_weights,
                      augment_schedule,
                      augmentation_mode
                      )

    if mode in ('test', 'full'):
        result = test(model_name,
                      test_images_path,
                      test_segs_path,
                      seg_ext,
                      class_path,
                      image_size,
                      n_classes,
                      test_batch_size,
                      checkpoint_path,
                      test_mode)

    print(datetime.now() - start)
