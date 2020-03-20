#!/usr/bin/env python3


import os
import cv2
import shutil

from utils import load_pickle_file

allowed_width = 256
allowed_height = 192 

images, labels = load_pickle_file("pickles/image_data_normalized.p")

if not os.path.exists("test_images"):
    os.makedirs("test_images")

for img_path in images:
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img,(allowed_width,allowed_height))
    
    parts = img_path.split("/")
    save_path = os.path.join("test_images","_".join(parts[-2:]))
    cv2.imwrite(save_path,resized_img)