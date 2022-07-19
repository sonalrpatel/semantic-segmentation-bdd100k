from collections import OrderedDict
from tensorflow import keras

import configs
from utils.helpers import *


class DataLoaderError(Exception):
    pass


# Prepare pairs
def get_pairs_from_paths(
        images_path, segs_path, seg_ext
):
    """ Find all the images from the images_path directory and the segmentation images 
        from the segs_path directory while checking integrity of dataloader """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]

    imgs = []
    segs = []
    for type in ACCEPTABLE_IMAGE_FORMATS:
        imgs += list(images_path.glob("*" + type))
    for type in ACCEPTABLE_SEGMENTATION_FORMATS:
        segs += list(segs_path.glob("*" + type))

    # Compare size of train and segmentation dataloader
    assert len(imgs) == len(segs), "No of Train images and label mismatch"

    # Sort the lists
    sorted(imgs), sorted(segs)

    # Match image and segmentation folders
    # Pair an image and its segmentation mask
    img_seg_pairs = []
    for img in tqdm(imgs):
        assert segs_path / (img.stem + seg_ext + ".png") in segs, "{img} not there in segmentation folder"
        img_seg_pairs.append((img, segs_path / (img.stem + seg_ext + ".png")))

        # if len(img_seg_pairs) > 100:
        #     break

    return img_seg_pairs  # (image and segmentation) pairs


# Verify dataset
def verify_segmentation_dataset(
        images_path, segs_path, seg_ext, n_classes, check_details=False, show_all_errors=False
):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path, seg_ext)
        if not len(img_seg_pairs):
            print("Couldn't load any dataloader from images_path: {0} and segmentations path: {1}"
                  .format(images_path, segs_path))
            return False

        return_value = True
        if check_details:
            for image_full_path, seg_full_path in tqdm(img_seg_pairs):
                img = cv2.imread(image_full_path)
                seg = cv2.imread(seg_full_path)
                # Check dimensions match
                if not img.shape == seg.shape:
                    return_value = False
                    print(
                        "The size of image {0} and its segmentation {1} doesn't match (possibly the files are corrupt)."
                            .format(image_full_path, seg_full_path))
                    if not show_all_errors:
                        break
                else:
                    max_pixel_value = np.max(seg[:, :, 0])
                    if max_pixel_value >= n_classes:
                        return_value = False
                        print("The pixel values of the segmentation image {0} violating range [0, {1}]. "
                              "Found maximum pixel value {2}"
                              .format(seg_full_path, str(n_classes - 1), max_pixel_value))
                        if not show_all_errors:
                            break

        if return_value:
            print("Dataset verified!")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during dataloader loading\n{0}".format(str(e)))
        return False


# Visualize dataset
def visualize_segmentation_dataset(
        images_path, segs_path, seg_ext, do_augment=False,
        image_size=None, augment_name="aug_all", custom_aug=None):
    try:
        # Get image-segmentation pairs
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path, seg_ext)

        print("Please press any key to display the next image")
        for im_fn, seg_fn in img_seg_pairs:
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)

            print("Found the following classes in the segmentation image:", np.unique(seg))

            if image_size is not None:
                img = cv2.resize(img, image_size)
                seg = cv2.resize(seg, image_size)

            print("Please press any key to display the next image")
            cv2.imshow("img", img)
            cv2.imshow("seg_img", seg)
            cv2.waitKey()
    except DataLoaderError as e:
        print("Found error during dataloader loading\n{0}".format(str(e)))
        return False


# Custom dataloader generator
class DataGenerator(keras.utils.Sequence):
    """
    Generates dataloader for Keras
    Reference:
        https://github.com/bdd100k/bdd100k/blob/a093762959e22ab4ed178b455897a801b96cc908/bdd100k/label/label.py
        https://www.kaggle.com/solesensei/solesensei_bdd100k
    """
    _color_encoding38 = OrderedDict([
        ('unlabeled-egovehicle-static', (0, 0, 0)),
        ('dynamic', (111, 74, 0)),
        ('ground', (81, 0, 81)),
        ('parking', (250, 170, 160)),
        ('rail_track', (230, 150, 140)),
        ('road', (128, 64, 128)),
        ('sidewalk', (244, 35, 232)),
        ('bridge', (150, 100, 100)),
        ('building', (70, 70, 70)),
        ('fence', (190, 153, 153)),
        ('garage', (180, 100, 180)),
        ('guard_rail', (180, 165, 180)),
        ('tunnel', (150, 120, 90)),
        ('wall', (102, 102, 156)),
        ('banner', (250, 170, 100)),
        ('billboard', (220, 220, 250)),
        ('lane_divider', (255, 165, 0)),
        ('pole', (153, 153, 153)),
        ('polegroup', (153, 153, 153)),
        ('street', (220, 220, 100)),
        ('traffic_cone', (255, 70, 0)),
        ('traffic_device', (220, 220, 220)),
        ('traffic_light', (250, 170, 30)),
        ('traffic_sign', (220, 220, 0)),
        ('traffic_sign_frame', (250, 170, 250)),
        ('terrain', (152, 251, 152)),
        ('vegetation', (107, 142, 35)),
        ('sky', (70, 130, 180)),
        ('person', (220, 20, 60)),
        ('rider', (255, 0, 0)),
        ('bicycle', (119, 11, 32)),
        ('bus', (0, 60, 100)),
        ('car', (0, 0, 142)),
        ('caravan', (0, 0, 90)),
        ('motorcycle', (0, 0, 230)),
        ('trailer', (0, 0, 110)),
        ('train', (0, 80, 100)),
        ('truck', (0, 0, 70))
    ])

    def __init__(self, images_path, segs_path, seg_ext, class_labels,
                 batch_size=16, shuffle=True, dim=(224, 224, 3),
                 aug_schedule=False, aug_mode=None):
        # TODO: if aug_schedule and aug_mode is not None, raise warning
        # Initialization
        self.dim = dim
        self.pair = get_pairs_from_paths(images_path, segs_path, seg_ext)
        self.indexes = np.arange(len(self.pair))
        self.class_labels = class_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug_schedule = aug_schedule
        self.aug_mode = aug_mode

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of dataloader when the batch corresponding to a given index
        is called, the generator executes the __getitem__ method to generate it.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate dataloader
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Shuffle indexes after each epoch
        Set augmentation mode as per global AUGMENTATION_MODE
        """
        # Shuffle the dataset
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Update aug_mode as per scheduling on plateau which is checked in callbacks
        if self.aug_schedule:
            self.aug_mode = configs.AUGMENTATION_MODE

    def __data_generation(self, list_IDs_temp):
        """Generates dataloader containing batch_size samples
        """
        # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_segs = list()

        # Generate dataloader
        for i in list_IDs_temp:
            img = img_to_array(load_img(self.pair[i][0], target_size=self.dim))
            seg = img_to_array(load_img(self.pair[i][1], target_size=self.dim))

            # org_h = cv2.hconcat([img, seg])

            # Augment the image, seg_mask
            if self.aug_mode is not None:
                augmented = self.aug_mode(image=img, mask=seg)
                img = augmented['image']
                seg = augmented['mask']

            # aug_h = cv2.hconcat([img, seg])
            # org_aug = cv2.vconcat([org_h, aug_h])
            # cv2.imshow("org_aug", org_aug / 255)

            # Normalise the image and one hot encode the seg_mask
            img = img / 255.
            seg = one_hot_image(seg, self.class_labels)

            batch_imgs.append(img)
            batch_segs.append(seg)

        return np.array(batch_imgs), np.array(batch_segs)

    def get_class_rgb_encoding(self):
        """
        Returns:
            An ordered dictionary encoding for pixel value, class name, and
            class color.
        """
        return self._color_encoding38.copy()
