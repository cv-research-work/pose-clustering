#!/usr/bin/env python3

import cv2 
import pickle

import matplotlib.pyplot as plt 

def load_pickle_file(pickle_file):
    img_data = pickle.load( open( pickle_file, "rb" ) )
    return img_data["images"],img_data["keypoints"]

## Load image and display keypoints
def show_img_with_keypoints(image, keypoints,label_names):
    def draw_circle(image, point):
        x,y = point[0],point[1]
        print(x,y)
        image = cv2.circle(image,(int(x),int(y)),20,(20,0,255),cv2.FILLED,3,0)
        return image

    img = cv2.imread(image)
    
    for idx, point in enumerate(keypoints):
        x,y = point[0],point[1]
        if x is None and y is None:
            continue
        ## draw a circle
        label = label_names[idx]
        img = draw_circle(img,[x,y])
    img = img[...,::-1]
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.show()


def normalize_keypoints(point_coordinate, max_bbox, min_bbox):
    return (point_coordinate - min_bbox) / (max_bbox - min_bbox);
