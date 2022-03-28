import glob
import cv2
import skimage.io as io
import skimage.transform as trans
import numpy as np
import pylab as plt
import albumentations

from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, GridDropout, ColorJitter,
    RandomBrightnessContrast, RandomGamma, OneOf, Rotate, RandomSunFlare, Cutout,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, HueSaturationValue,
    RGBShift, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop
)

# References
# https://github.com/mjkvaak/ImageDataAugmentor.git
# https://www.kaggle.com/meaninglesslives/nested-unet-with-efficientnet-encoder

image_size = (192, 192, 3)

AUGMENTATIONS_TRAIN_SOFT = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(rotate_limit=20, shift_limit=0.07, scale_limit=0.2, p=0.3),
    OneOf([
        RandomSizedCrop(min_max_height=(96, 160), height=image_size[0], width=image_size[1]),
        Cutout(num_holes=4)
    ], p=0.2)
], p=1)

AUGMENTATIONS_TRAIN_HARD = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(rotate_limit=40, shift_limit=0.1, scale_limit=0.4, p=0.5),
    OneOf([
        RandomSizedCrop(min_max_height=(96, 144), height=image_size[0], width=image_size[1]),
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


def square_image(img, random=None):
    """ Square Image
    Function that takes an image (ndarray),
    gets its maximum dimension,
    creates a black square canvas of max dimension
    and puts the original image into the
    black canvas's center
    If random [0, 2] is specified, the original image is placed
    in the new image depending on the coefficient,
    where 0 - constrained to the left/up anchor,
    2 - constrained to the right/bottom anchor
    """
    size = max(img.shape[0], img.shape[1])
    new_img = np.zeros((size, size), np.float32)
    ax, ay = (size - img.shape[1]) // 2, (size - img.shape[0]) // 2

    if random and not ax == 0:
        ax = int(ax * random)
    elif random and not ay == 0:
        ay = int(ay * random)

    new_img[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
    return new_img


def reshape_image(img, target_size):
    """ Reshape Image
    Function that takes an image
    and rescales it to target_size
    """
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)
    return img


def normalize_mask(mask: object) -> object:
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask


def show_image(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_width)
        y = random.randint(0, image.shape[0] - crop_height)

        if len(label.shape) == 3:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width, :]
        else:
            return image[y:y + crop_height, x:x + crop_width, :], label[y:y + crop_height, x:x + crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
            crop_height, crop_width, image.shape[0], image.shape[1]))


def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0 * args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1 * args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                     flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]),
                                      flags=cv2.INTER_NEAREST)

    return input_image, output_image
