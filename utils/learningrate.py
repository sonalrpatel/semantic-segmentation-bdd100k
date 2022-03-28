# importing the required module
import matplotlib.pyplot as plt
import math


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


# epoch = list(range(1, 41))
# lr = [lr_const_exp_decay_(epoch) for epoch in epoch]
# plt.plot(epoch, lr)
# plt.show()
#
# exit(0)
