import os
import warnings

import math
import numpy as np
import scipy.signal
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import backend as K

import configs
from configs import *


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses = []
        self.val_loss = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")


class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self, decay_rate, verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate = decay_rate
        self.verbose = verbose
        self.learning_rates = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.learning_rate) * self.decay_rate
        K.set_value(self.model.optimizer.learning_rate, learning_rate)
        if self.verbose > 0:
            print('\nSetting learning rate to %s.' % learning_rate)


class AugmentationScheduleOnPlateau(keras.callbacks.Callback):
    """
    Schedule augmentation on Training plateau
    Reference:
        https://www.kaggle.com/c/inclusive-images-challenge/discussion/72450
        https://keras.io/guides/writing_your_own_callbacks/

    Schedule Augmentation when the val_mean_iou stops increasing.
    Arguments:
      patience: Number of epochs to wait after max has been hit. Then after,
      apply soft augmentation for few epochs, and hard augmentation.
    """

    def __init__(self, init_delay_epoch=10, patience=7, min_delta=0.01,
                 reset_lr=None, verbose=0):
        super(AugmentationScheduleOnPlateau, self).__init__()
        self.init_delay_epoch = init_delay_epoch
        self.patience = patience
        self.wait = 0
        self.best = 0
        self.min_delta = min_delta
        self.aug_mode = configs.AUGMENTATION_MODE
        self.reset_lr = reset_lr
        self.verbose = verbose

    @staticmethod
    def get_aug_mode_no(key):
        aug_mode_dict = {
            None: 0,
            AUGMENTATIONS_TRAIN_SOFT: 1,
            AUGMENTATIONS_TRAIN_HARD: 2
        }

        return aug_mode_dict[key]

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        current = logs.get("val_mean_iou")

        if epoch <= self.init_delay_epoch:
            self.best = current
        else:
            if self.aug_mode != AUGMENTATIONS_TRAIN_HARD:
                if np.greater(current, self.best + self.min_delta):
                    self.best = current
                    self.wait = 0
                else:
                    self.wait += 1

        if self.wait >= self.patience:
            # reset wait to start counting again
            self.wait = 0

            if self.aug_mode is None:
                self.aug_mode = AUGMENTATIONS_TRAIN_SOFT
            elif self.aug_mode == AUGMENTATIONS_TRAIN_SOFT:
                self.aug_mode = AUGMENTATIONS_TRAIN_HARD
            else:
                pass

            configs.AUGMENTATION_MODE = self.aug_mode

            if self.verbose > 0:
                print("\n{} augmentation is applied".format(self.aug_mode))

            # set the learning rate to the reset_lr value when aug_mode changes
            if self.reset_lr is not None:
                K.set_value(self.model.optimizer.learning_rate, self.reset_lr)
                if self.verbose > 0:
                    print('\nSetting learning rate to %s.' % self.reset_lr)

        logs = logs or {}
        logs['aug_mode'] = self.get_aug_mode_no(self.aug_mode)


def lr_static_(epoch):
    initial_lr = 0.001
    return initial_lr


def lr_step_decay_(epoch):
    initial_lr = 0.001
    drop_rate = 0.5
    epochs_drop = 10
    return initial_lr * math.pow(drop_rate, math.floor(epoch / epochs_drop))


def lr_exp_decay_(epoch):
    initial_lr = 0.001
    k = 0.07
    return initial_lr * math.exp(-k * epoch)


def lr_const_exp_decay_(epoch):
    max_epoch = 40
    initial_lr = 0.001
    k = 0.1
    if epoch < max_epoch/2:
        return initial_lr
    else:
        return initial_lr * math.exp(k * (max_epoch/2 - epoch))


def lr_exp_decay_reset_(epoch):
    initial_lr = 0.001  # 0.005
    drop_rate = 0.5
    epochs_drop = 10
    k = 0.2
    mod_init_lr = initial_lr * math.pow(drop_rate, math.floor(epoch / epochs_drop))
    return mod_init_lr * math.exp(-k * (epoch % 10))


def lr_cosine_(epoch):
    initial_lr = 0.001
    return (initial_lr / 2) * (1 + math.cos(0.4 * epoch))
