"""
Created on Sat Apr 24 23:08:34 2021

@author: sonal
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path as path
from tqdm import tqdm

from train import train

images_path = path("C:/Users/sonal/Google Drive/10_Python/1_0_Datasets/camvid/train/images/all")
segs_path = path("C:/Users/sonal/Google Drive/10_Python/1_0_Datasets/camvid/train/labels/all")
seg_ext = "_L"
class_path = path("C:/Users/sonal/Google Drive/10_Python/1_0_Datasets/camvid/class_dict.csv")

imgs = list(images_path.glob("*.png"))
segs = list(segs_path.glob("*.png"))


# Compare size of train and segmentation dataloader
assert len(imgs) == len(segs), "No of Train images and label mismatch"


# Sort the lists
sorted(imgs), sorted(segs);


# Match and pair an image and its segmentation
def make_pair(imgs, segs_path, seg_ext):
    img_seg_pairs = []
    for img in tqdm(imgs):
        assert segs_path / (img.stem + seg_ext + ".png") in segs, "{img} not there in segmentation folder"
        img_seg_pairs.append((img, segs_path / (img.stem + seg_ext + ".png")))
        
    return img_seg_pairs


img_seg_pairs = make_pair(imgs, segs_path, seg_ext)


filename, file_extension = os.path.splitext(class_path)
if not file_extension == ".csv":
    print("File is not a CSV!")

if file_extension == ".csv":
    class_dict = pd.read_csv(class_path)
elif file_extension == ".xlxs":
    class_dict = pd.read_excel(class_path)
else:
    print("File format not supported")
    
class_names = []
rgb_values = []
for index, item in class_dict.iterrows():
    class_names.append(item[0])
    try:
        rgb_values.append(np.array([item['red'], item['green'], item['blue']]))
    except:
        try:
            rgb_values.append(np.array([item['k'], item['g'], item['b']]))
        except:
            print("Column names are not appropiate")
            break