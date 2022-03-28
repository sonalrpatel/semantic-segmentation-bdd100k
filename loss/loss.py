from tensorflow.python.keras.losses import categorical_crossentropy
from utils.utils_metric import MeanIoU
from utils.utils_metric import dice_coef


class LossFunc(object):
    """""
    Calculates loss
    """""

    def __init__(self, num_classes):
        super(LossFunc, self).__init__()
        self.miou_metric = MeanIoU(num_classes)

    def iou_loss(self, y_true, y_pred):
        loss = 1 - self.miou_metric.mean_iou(y_true, y_pred)
        return loss

    def dice_loss(self, y_true, y_pred):
        loss = 1 - dice_coef(y_true, y_pred)
        return loss

    def CEIoU_loss(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred) + self.iou_loss(y_true, y_pred)
        return loss

    def CEDice_loss(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss
