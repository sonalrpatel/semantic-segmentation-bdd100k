import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping

from model.unet_adv import unet_adv
from model.unet import *
from model.pspnet import *
from model.deeplabv3 import *
from model.fpn import *

from loss.loss import LossFunc
from dataloader.dataloader import *
from utils.utils_metric import MeanIoU
from utils.learningrate import *
from configs import *


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


def print_info(
        checkpoint_path,
        train_generator,
        val_generator,
        initial_epoch
):
    """
    Print information about model training
    """
    image_batch, label_batch = train_generator[0]
    num_classes = label_batch[0].shape[-1]
    print("\n")
    print("--> Model under trail: {}".format(checkpoint_path.split('/')[-2]))
    print("--> Starting with initial_epoch: {}".format(initial_epoch))
    print("--> Training batches: {}".format(len(train_generator)))
    print("--> Validation batches: {}".format(len(val_generator)))
    print("--> Image size: {}".format(image_batch.shape))
    print("--> Label size: {}".format(label_batch.shape))
    print("--> No. of classes: {}".format(num_classes))
    print("\n")


# =======================================================
# Train the model
# =======================================================
def _main():
    # =======================================================
    #   Be sure to modify classes_path before training so that it corresponds to your own dataset
    # =======================================================
    classes_path = PATH_CLASSES

    # =======================================================
    #   When weight_path = '', the weights of the entire model are not loaded
    # =======================================================
    weight_path = PATH_WEIGHT


    global initial_epoch, model, val_generator
    assert (n_classes is not None), "Please provide the n_classes"
    assert (image_size is not None), "Please provide the image_size"
    assert (model_name is not None), "Please provide the model name"
    assert (optimizer_name is not None), "Please specify the optimizer"

    # Verify dataset
    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(images_path, segs_path, seg_ext, n_classes)
        assert verified

        if validate_using == "val":
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images_path, val_segs_path, seg_ext, n_classes)
            assert verified

    # Create generators
    train_generator = DataGenerator(images_path, segs_path, seg_ext, class_path, batch_size,
                                    augment_schedule=augment_schedule, augmentation_mode=augmentation_mode,
                                    dim=image_size, image_enhance=False)

    if validate_using == "val":
        val_generator = DataGenerator(val_images_path, val_segs_path, seg_ext, class_path, val_batch_size,
                                      dim=image_size, image_enhance=False)

    # Initialize accuracy metric - miou
    miou_metric = MeanIoU(n_classes)

    if not resume_checkpoint:
        # Build model
        model_cfg = (n_classes, image_size, encoder, weights, model_name)
        if "unet_adv" in model_name:
            model = unet_adv(model_cfg)
        elif "unet" in model_name:
            model = unet(model_cfg)
        elif "pspnet" in model_name:
            model = pspnet(model_cfg)
        elif "deeplabv3" in model_name:
            model = deeplabv3(model_cfg)
        elif "fpn" in model_name:
            model = fpn(model_cfg)
        else:
            raise "model name is not provided"

        # Model summary
        model.summary()

        # Save model plot
        if checkpoint_path is not None:
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            plot_model(model, to_file=checkpoint_path + 'model_plot.png', show_shapes=True, show_layer_names=True)

        # Loss function
        losses = LossFunc(n_classes)

        if loss_type == "iou":
            loss_k = losses.iou_loss
        elif loss_type == "dice":
            loss_k = losses.dice_loss
        elif loss_type == "ceiou":
            loss_k = losses.CEIoU_loss
        elif loss_type == "cedice":
            loss_k = losses.CEDice_loss
        else:
            loss_k = "categorical_crossentropy"

        # Compile
        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy', miou_metric.mean_iou])

        # Load weights
        initial_epoch = 0
        if model_weights is not None and len(model_weights) > 0:
            print("Loading weights from ", model_weights)
            model.load_weights(model_weights)

    # Resume checkpoint
    if resume_checkpoint is True and checkpoint_path is not None:
        # Find the latest checkpoint
        latest_checkpoint = sorted(glob.glob(checkpoint_path + "*.hdf5"))[-1]

        # Load model from checkpoint_path
        if latest_checkpoint is not None:
            initial_epoch = int(latest_checkpoint.split('.')[0][-4:-2])
            print("Loading the complete model from latest checkpoint ", latest_checkpoint)

            model = load_model(latest_checkpoint,
                               custom_objects={'mean_iou': miou_metric.mean_iou}
                               )

    # Print information about the training
    print_info(checkpoint_path, train_generator, val_generator, initial_epoch)

    # Callbacks
    if callback_fn is None:
        filepath_k = checkpoint_path + model_name + "_{epoch:02d}.hdf5"
        callback_fn = [ModelCheckpoint(filepath=filepath_k, save_weights_only=True, verbose=True)]

    if callback_fn == "all":
        filepath_k = checkpoint_path + model_name + "_{epoch:02d}-{val_mean_iou:.3f}.hdf5"
        log_dir_k = checkpoint_path + "/logs/"
        monitor_k = 'val_mean_iou'
        mode_k = 'max'

        # Augmentation scheduling callback
        ag = AugmentationScheduleOnPlateau(patience=7, init_delay_epoch=10)

        # LearningRateScheduler
        lrs = LearningRateScheduler(lr_const_exp_decay_, verbose=True)

        # CustomLearningRateScheduler -> LearningRatePlanner
        # lrs = LearningRatePlanner(init_lr=0.001, patience=5, factor=0.5, reset_on_aug_start=True)

        # Reduce LR on Plateau
        # lrs = ReduceLROnPlateau(mode=mode_k, monitor=monitor_k, factor=0.5, patience=5, verbose=True, min_lr=0.0001)

        # Checkpoint callback - save the best model
        mc = ModelCheckpoint(mode=mode_k, filepath=filepath_k, monitor=monitor_k, save_best_only='True', verbose=True)

        # Early stopping
        es = EarlyStopping(mode=mode_k, monitor=monitor_k, min_delta=0.01, patience=20, verbose=True)

        # TensorBoard callback
        tb = TensorBoard(log_dir=log_dir_k, histogram_freq=0, write_graph=True, write_images=False)

        # CSVLogger
        cv = CSVLogger(log_dir_k + "log.csv", append=True, separator=',')

        if not augment_schedule:
            callback_fn = [lrs, mc, tb, cv]
        else:
            callback_fn = [ag, lrs, mc, tb, cv]

        if validate_using == "using_val" and tensorboard_viz_use is True:
            # Tensorboard callback that displays a random sample with respective target and prediction
            tensorboard_viz = TensorBoardPrediction(
                val_generator, val_generator.get_class_rgb_encoding(), log_dir=log_dir_k
            )
            callback_fn = [lrs, mc, es, tb, cv, tensorboard_viz]

    # Train the model
    if validate_using == "using_val":
        history = model.fit(train_generator, steps_per_epoch=train_generator.__len__(),
                            validation_data=val_generator, validation_steps=val_generator.__len__(),
                            epochs=epochs, callbacks=callback_fn, initial_epoch=initial_epoch)
    elif validate_using == "using_train":
        history = model.fit(train_generator, steps_per_epoch=train_generator.__len__(),
                            validation_split=0.2,
                            epochs=epochs, callbacks=callback_fn, initial_epoch=initial_epoch)
    else:
        history = model.fit(train_generator, steps_per_epoch=train_generator.__len__(),
                            epochs=epochs, callbacks=callback_fn, initial_epoch=initial_epoch)

    # Plot result
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
