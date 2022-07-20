import os
import glob
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from model.unet import *
from model.pspnet import *
from model.deeplabv3 import *
from model.fpn import *

import configs
from loss.loss import LossFunc
from dataloader.dataloader import *
from utils.utils_metric import MeanIoU
from utils.callbacks import *

# =======================================================
# Set a seed value
# =======================================================
seed_value = 121

# =======================================================
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
# =======================================================
os.environ['PYTHONHASHSEED'] = str(seed_value)

# =======================================================
# 2. Set `python` built-in pseudo-random generator at a fixed value
# =======================================================
random.seed(seed_value)

# =======================================================
# 3. Set `numpy` pseudo-random generator at a fixed value
# =======================================================
np.random.seed(seed_value)

# =======================================================
# 4. Set `tensorflow` pseudo-random generator at a fixed value
# =======================================================
tf.random.set_seed(seed_value)
print(tf.__version__)


# =======================================================
# Train the model
# =======================================================
def _main():
    # =======================================================
    #   The size of the input shape must be a multiple of 32
    # =======================================================
    image_size = IMAGE_SIZE

    # =======================================================
    #   Be sure to modify classes_path before training so that it corresponds to your own dataset
    # =======================================================
    classes_path = PATH_CLASSES

    # =======================================================
    #   Model configurations
    # =======================================================
    model_name = MODEL_NAME
    encoder = MODEL_ENCODER
    encoder_weights = MODEL_ENCODER_WEIGHTS
    optimizer = MODEL_OPTIMIZER
    loss_type = MODEL_LOSS

    # =======================================================
    #   When model_weights = '', the weights of the entire model are not loaded
    # =======================================================
    model_weights = PATH_WEIGHTS

    # =======================================================
    #   Augmentation settings
    # =======================================================
    aug_schedule = AUGMENTATION_SCHEDULE
    aug_mode = AUGMENTATION_MODE

    # =======================================================
    #   Training settings
    # =======================================================
    train_images_path = DIR_TRAIN_IMG
    train_segs_path = DIR_TRAIN_SEG
    seg_name_ext = ''
    init_epoch = TRAIN_FREEZE_INIT_EPOCH
    freeze_end_epoch = TRAIN_FREEZE_END_EPOCH
    unfreeze_end_epoch = TRAIN_UNFREEZE_END_EPOCH
    freeze_batch_size = TRAIN_FREEZE_BATCH_SIZE
    unfreeze_batch_size = TRAIN_UNFREEZE_BATCH_SIZE
    freeze_lr = TRAIN_FREEZE_LR
    unfreeze_lr = TRAIN_UNFREEZE_LR
    verify_dataset = VERIFY_DATASET

    # =======================================================
    #   Validation settings
    # =======================================================
    val_images_path = DIR_VAL_IMG
    val_segs_path = DIR_VAL_SEG
    val_batch_size = VAL_BATCH_SIZE
    val_using = VAL_VALIDATION_USING
    val_split = VAL_VALIDATION_SPLIT

    # =======================================================
    #   Checkpoint settings
    # =======================================================
    log_dir = LOG_DIR
    checkpoint_path = LOG_DIR2
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    checkpoint_resume = TRAIN_FROM_CHECKPOINT

    # =======================================================
    #   Get classes and details
    # =======================================================
    num_classes, class_names, class_labels = get_class_info(classes_path)

    # =======================================================
    #   Verify dataset
    # =======================================================
    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images_path, train_segs_path, seg_name_ext, num_classes)
        assert verified

        if val_using == "val":
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images_path, val_segs_path, seg_name_ext, num_classes)
            assert verified

    # =======================================================
    #   Initialize evaluation metric (accuracy - mean_iou)
    # =======================================================
    mean_iou = MeanIoU(num_classes)

    # =======================================================
    #   Build the model
    # =======================================================
    if not checkpoint_resume:
        # TODO: clean up and optimise the model building part
        model_cfg = (num_classes, image_size, encoder, encoder_weights, model_name)
        if "unet" in model_name:
            model = unet(model_cfg)
        elif "pspnet" in model_name:
            model = pspnet(model_cfg)
        elif "deeplabv3" in model_name:
            model = deeplabv3(model_cfg)
        elif "fpn" in model_name:
            model = fpn(model_cfg)
        else:
            raise "Model name is not provided"

        # =======================================================
        #   Model summary and Plot model
        # =======================================================
        model.summary()
        plot_model(model, to_file=checkpoint_path + str(model_name) + '_model_plot.png',
                   show_shapes=True, show_layer_names=True)

        # =======================================================
        #   Loss function
        # =======================================================
        loss_fn = LossFunc(num_classes)

        if loss_type == "iou":
            loss = loss_fn.iou_loss
        elif loss_type == "dice":
            loss = loss_fn.dice_loss
        elif loss_type == "ce_iou":
            loss = loss_fn.CEIoU_loss
        elif loss_type == "ce_dice":
            loss = loss_fn.CEDice_loss
        else:
            loss = "categorical_crossentropy"

        # =======================================================
        #   Model compile
        # =======================================================
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy', mean_iou.mean_iou])

        # =======================================================
        #   Load weights
        # =======================================================
        if model_weights is not None:
            print("Loading weights from ", model_weights)
            model.load_weights(model_weights)

    # =======================================================
    #   Load complete model from a checkpoint and Resume training
    # =======================================================
    if checkpoint_resume:
        # Find the latest checkpoint
        latest_checkpoint = sorted(glob.glob(checkpoint_path + "*.hdf5"))[-1]

        # Load complete model from latest_checkpoint
        if latest_checkpoint is not None:
            init_epoch = int(latest_checkpoint.split('.')[0][-4:-2])
            print("Loading the complete model from latest checkpoint ", latest_checkpoint)

            model = load_model(latest_checkpoint,
                               custom_objects={'mean_iou': mean_iou.mean_iou}
                               )

    # =======================================================
    #   Callbacks
    # =======================================================
    filepath = checkpoint_path + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    monitor = 'val_loss'
    mode = 'min'

    tb_logging = TensorBoard(log_dir)
    csv_logger = CSVLogger(log_dir + "log.csv", append=True, separator=',')
    reduce_lr = ExponentDecayScheduler(decay_rate=0.94, verbose=True)
    checkpoint = ModelCheckpoint(filepath, monitor=monitor, mode=mode,
                                 save_weights_only=True, save_best_only=True, period=10, verbose=True)
    early_stopping = EarlyStopping(monitor=monitor, mode=mode, min_delta=0, patience=10, verbose=True)
    loss_history = LossHistory(log_dir)

    callbacks_all = [tb_logging, csv_logger, checkpoint, reduce_lr, early_stopping, loss_history]

    if aug_schedule:
        aug_change = AugmentationScheduleOnPlateau(patience=7, init_delay_epoch=10)
        callbacks_all.append(aug_change)

    # =======================================================
    #   Annotation pairs
    # =======================================================
    train_pairs = get_pairs_from_paths(train_images_path, train_segs_path, seg_name_ext)
    if val_using == "VAL":
        val_pairs = get_pairs_from_paths(val_images_path, val_segs_path, seg_name_ext)
    if val_using == "TRAIN":
        val_pairs = random.sample(train_pairs, int(len(train_pairs) * val_split))
        train_pairs = [line for line in train_pairs if line not in val_pairs]

    # =======================================================
    #   Create data generators
    # =======================================================
    train_generator = DataGenerator(train_pairs, class_labels, freeze_batch_size, dim=image_size,
                                    aug_schedule=aug_schedule, aug_mode=aug_mode)

    if val_using == "VAL" or val_using == "TRAIN":
        val_generator = DataGenerator(val_pairs, class_labels, val_batch_size, dim=image_size)

    # =======================================================
    #   Train the model
    # =======================================================
    if val_using == "VAL" or val_using == "TRAIN":
        print("Training with {} train samples and validating with {} val samples from {}."
              .format(len(train_pairs), len(val_pairs), val_using))
        history = model.fit(train_generator, steps_per_epoch=train_generator.__len__(),
                            validation_data=val_generator, validation_steps=val_generator.__len__(),
                            epochs=freeze_end_epoch, callbacks=callbacks_all, initial_epoch=init_epoch)
    else:
        print("Training with {} train samples without validation.".format(len(train_pairs)))
        history = model.fit(train_generator, steps_per_epoch=train_generator.__len__(),
                            epochs=freeze_end_epoch, callbacks=callbacks_all, initial_epoch=init_epoch)

    # =======================================================
    #   Plot result
    # =======================================================
    plt.title("loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="val")
    plt.legend(loc="best")
    plt.savefig(checkpoint_path + model_name + '_loss.png')

    plt.gcf().clear()
    plt.title("mean_iou")
    plt.plot(history.history["mean_iou"], color="r", label="train")
    plt.plot(history.history["val_mean_iou"], color="b", label="val")
    plt.legend(loc="best")
    plt.savefig(checkpoint_path + model_name + '_mean_iou.png')

    return model


if __name__ == '__main__':
    _main()
