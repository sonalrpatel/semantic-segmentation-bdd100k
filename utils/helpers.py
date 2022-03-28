import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from pathlib import Path as path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Get class names and labels
def get_class_info(class_path):
    """
    Retrieve the class names and RGB values for the selected dataset.
    Must be in CSV or XLXS format!

    # Arguments
        class_path: The file path of the class dictionairy

    # Returns
        Two lists: one for the class names and the other for the class label
    """
    global class_dict
    filename, file_extension = os.path.splitext(class_path)

    if file_extension == ".csv":
        class_dict = pd.read_csv(class_path)
    elif file_extension == ".xlsx":
        class_dict = pd.read_excel(class_path)
    else:
        print("Class dictionary file format not supported")
        exit(1)

    class_names = []
    class_labels = []
    for index, item in class_dict.iterrows():
        class_names.append(item[0])
        try:
            class_labels.append(np.array([item['red'], item['green'], item['blue']]))
        except:
            try:
                class_labels.append(np.array([item['r'], item['g'], item['b']]))
            except:
                print("Column names are not appropiate")
                break

    return class_names, class_labels


# Get labeled segmentation mask
def label_segmentation_mask(seg, class_labels):
    """
    Given a 3D (W, H, depth=3) segmentation mask, prepare a 2D labeled segmentation mask

    # Arguments
        seg: The segmentation mask where each cell of depth provides the r, g, and b values
        class_labels

    # Returns
        Labeled segmentation mask where each cell provides its label value
    """
    seg = seg.astype("uint8")

    # returns a 2D matrix of size W x H of the segmentation mask
    label = np.zeros(seg.shape[:2], dtype=np.uint8)

    for i, rgb in enumerate(class_labels):
        label[(seg == rgb).all(axis=2)] = i

    return label


def one_hot_image(seg, class_labels):
    """
    Convert a segmentation mask label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        seg: The 3D array segmentation mask
        class_labels

    # Returns
        A 3D array with the same width and height as the input, but
        with a depth size of num_classes
    """
    num_classes = len(class_labels)                         # seg dim = H*W*3
    label = label_segmentation_mask(seg, class_labels)      # label dim = H*W
    one_hot = to_categorical(label, num_classes)            # one_hot dim = H*W*N

    return one_hot


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,1])

    for i in range(0, w):
        for j in range(0, h):
            index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
            x[i, j] = index

    x = np.argmax(image, axis=-1)
    return x


def make_prediction(model, img=None, img_path=None, shape=None):
    """
    Predict the hot encoded categorical label from the image.
    Later, convert it numerical label.
    """
    if img is not None:                         # dim = H*W*3
        img = np.expand_dims(img, axis=0)       # dim = 1*H*W*3
    if img_path is not None:
        img = img_to_array(load_img(img_path, target_size=shape)) / 255.
        img = np.expand_dims(img, axis=0)       # dim = 1*H*W*3
    label = model.predict(img)                  # dim = 1*H*W*N
    label = np.argmax(label[0], axis=2)         # dim = H*W

    return label


def form_color_mask(label, mapping):
    """
    Generate the color mask from the numerical label
    """
    h, w = label.shape                          # dim = H*W
    mask = np.zeros((h, w, 3), dtype=np.uint8)  # dim = H*W*3
    mask = mapping[label]
    mask = mask.astype(np.uint8)

    return mask


def count_pixels(seg_path, class_path):
    seg_path_list = list(seg_path.glob("*.png"))
    class_names, class_labels = get_class_info(class_path)
    class_list = [str(list(x)) for x in class_labels]
    df_class_count = pd.DataFrame(columns=class_list)

    # for each segmentation mask
    for enum, seg_p in tqdm(enumerate(seg_path_list)):
        seg = img_to_array(load_img(seg_p, target_size=(192, 192, 3)))
        seg_2d = [str(list(x.astype(int))) for x in seg.reshape(-1, seg.shape[2])]
        unq_cls_list = [str(list(x.astype(int))) for x in np.unique(seg.reshape(-1, seg.shape[2]), axis=0)]

        df_class_count.loc[enum] = [0] * len(class_names)

        # for each unique pixel
        for unq_cls in unq_cls_list:
            df_class_count.loc[enum, unq_cls] = seg_2d.count(unq_cls)

    df_class_count.columns = class_names
    df_class_count = pd.DataFrame(df_class_count.sum(axis=0)).T

    return df_class_count


def count_unique_pixels(seg_path):
    seg_path_list = list(seg_path.glob("*.png"))

    num_unique_pixels = []

    # for each segmentation mask
    for enum, seg_p in tqdm(enumerate(seg_path_list)):
        seg = img_to_array(load_img(seg_p, target_size=(192, 192, 3)))
        seg_2d = [str(list(x.astype(int))) for x in seg.reshape(-1, seg.shape[2])]
        unq_cls_list = [str(list(x.astype(int))) for x in np.unique(seg.reshape(-1, seg.shape[2]), axis=0)]

        num_unique_pixels.append(len(unq_cls_list))

    return num_unique_pixels


# primary_path = "C:/Users/sonal/Google Drive/10_Python/1_0_Datasets/bdd100k/seg"
# seg_path_train = path(primary_path + "/color_labels/train_")
# seg_path_test = path(primary_path + "/color_labels/test_")
# seg_path_val = path(primary_path + "/color_labels/val")
# class_path = path(primary_path + "/class_dict_41.xlsx")
#
# ds_cls_cnt_train = count_pixels(seg_path_train, class_path).loc[0]
# ds_cls_cnt_test = count_pixels(seg_path_test, class_path).loc[0]
# ds_cls_cnt_val = count_pixels(seg_path_val, class_path).loc[0]
# ds_cls_cnt = ds_cls_cnt_train.add(ds_cls_cnt_test).add(ds_cls_cnt_val)
# df_cls_cnt = pd.DataFrame(ds_cls_cnt)
#
# df_cls_cnt.plot.bar()
# plt.yscale("log")
# plt.show()
#
# list_num_unq_pxl_train = count_unique_pixels(seg_path_train, class_path)
