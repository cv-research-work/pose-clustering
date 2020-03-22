#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import pandas as pd
import shutil
import json
import numpy as np
import pickle
from PIL import Image
import cv2

import xml.etree.ElementTree as ET
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans, AgglomerativeClustering
import pdb 

import argparse
import utils

## use following keypoint labels
label_names = ["Top of the head",
                'Highest point on the back',
                "Left eye","Right eye",
                "Jaw",
                "Base of trunk",
                "Left elbow",
                "Right elbow",
                "Bottom of the belly",
                "Left knee",
                "Right knee",
                "Widest point of left ear",
                "Widest point of right ear",
                "Base of tail",
                "Bottom of the right backfoot",
                "Bottom of the right front foot",
                "Bottom of the left front foot",
                "Bottom of the left backfoot",
                "Base of right tusk",
                "Base of left tusk",
                ]

exclude_label_names = ["Jaw"]

def get_exclude_indexes():
    indexes = []
    for e in exclude_label_names:
        index = label_names.index(e)
        indexes.append(index)
    return indexes

def save_annotations_pickle_file(train_dict, filename=""):
        f = open(filename,"wb")
        pickle.dump(train_dict,f,-1)
        f.close()  

def get_artifacts_manual_anno(data_dir):
    images_list = []
    keypoints_list = []

    for folder in os.listdir(data_dir):
        if folder not in [".DS_Store"]:
            for f in os.listdir(os.path.join(data_dir, folder)):
                if f==".DS_Store":
                    continue
                filename, ext = os.path.splitext(f)

                ## dont process .MOV file
                if ext == ".MOV":
                    continue
                
                if not os.path.exists(os.path.join(data_dir,folder,filename)+".json"):
                    print(os.path.join(data_dir,folder,f) , "doesnt exists")
                    continue
                elif ext==".json":
                    with open(os.path.join(data_dir,folder,f)) as json_file:
                        data = json.load(json_file)

                    points = {}
                    if "shapes" not in data:
                        continue
                    
                    shapes = data["shapes"]
                    
                    img = cv2.imread(os.path.join(data_dir,folder,data["imagePath"]))
                    h,w, c = img.shape


                    ## get bounding box values
                    xml_path = os.path.join(data_dir,folder,filename+".xml")
                    if not os.path.exists(xml_path):
                        continue

                    if os.path.exists(xml_path):
                        xml_file = xml_path
                        
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        
                        for elem in root:
                            current_parent = elem.tag
                            
                            if current_parent == 'object':
                                current_parent = elem.tag
                                for subelem in elem:
                                    if subelem.tag=="bndbox":
                                        for e in subelem:
                                            i = e.tag
                                            
                                            if i=="xmin":
                                                xmin = float(e.text)
                                            elif i=="ymin":
                                                ymin = float(e.text)
                                            elif i=="xmax":
                                                xmax = float(e.text)
                                            else:
                                                ymax = float(e.text)
                    else:
                        print(xml_path + " doesnt exist")
                        exit()
                    for label in shapes:
                        x, y =   label["points"][0][0] , label["points"][0][1]
                        
                        new_x = utils.normalize_keypoints(x,xmax,xmin)
                        new_y = utils.normalize_keypoints(y,ymax,ymin)

                        point = [new_x,new_y]

                        ### in COCO,v = 0,not labeled;v = 1;labeled but invisible;v = 2,labeled and visible
                        if label["label"] in ['Top of the head','Highest point of the head']:
                            points[label_names.index('Top of the head')]=point

                        elif label["label"] in ['Highest point on the back', 'Highest point of the back','Highest point on back']:
                            points[label_names.index('Highest point on the back')] = point

                        elif label["label"] in ["Left eye","Left ey"]:
                            points[label_names.index('Left eye')] = point
                        
                        elif label["label"] in ["Right eye"]:
                            points[label_names.index('Right eye')] = point
                        
                        elif label["label"] in ["Jaw"]:
                            points[label_names.index("Jaw")] = point

                        elif label["label"] in ["Eyebrow of the face"]:
                            continue

                        elif label["label"] in ["Base of trunk","Base of the trunk"]:
                            points[label_names.index("Base of trunk")] = point
                        
                        elif label["label"] in ["End of trunk","End of the trunk"]:
                            continue

                        elif label["label"] in ["Base of tail","Base of the tail","Base of tal"]:
                            points[label_names.index("Base of tail")] = point

                        elif label["label"] in ["End of tail","End of the tail"]:
                            continue
                        
                        elif label["label"] in ["Left elbow"]:
                            points[label_names.index("Left elbow")] = point

                        elif label["label"] in ["Right elbow"]:
                            points[label_names.index("Right elbow")] = point

                        elif label["label"] in ["Bottom of the belly","Bottom of the bely","Bottom of belly"]:
                            points[label_names.index("Bottom of the belly")] = point

                        elif label["label"] in ["Left knee"]:
                            points[label_names.index("Left knee")] = point

                        elif label["label"] in ["Right knee"]:
                            points[label_names.index("Right knee")] = point

                        elif label["label"] in ["Base of left tusk","Base of the left tusk","Bottom of the left tusk"]:
                            points[label_names.index("Base of left tusk")] = point

                        elif label["label"] in ["Base of right tusk","Base of the right tusk"]:
                            points[label_names.index("Base of right tusk")] = point

                        elif label["label"] in ["End of right tusk","End of the right tusk"]:
                            continue

                        elif label["label"] in ["End of left tusk","End of the left tusk"]:
                            continue

                        elif label["label"] in ["Bottom of the right backfoot","Bottom of right backfoot","Right backfoot","Bottom of the right back foot"]:
                            points[label_names.index("Bottom of the right backfoot")] = point

                        elif label["label"] in ["Bottom of the left backfoot","Bottom of left backfoot","Left backfoot","Bottom of the left back foot"]:
                            points[label_names.index("Bottom of the left backfoot")] = point

                        elif label["label"] in ["Bottom of the left front foot","Bottom of left front foot"]:
                            points[label_names.index("Bottom of the left front foot")] = point

                        elif label["label"] in ["Bottom of the right front foot","Bottom of right front foot"]:
                            points[label_names.index("Bottom of the right front foot")] = point

                        elif label["label"] in ["Top of the left shoulder"]:
                            continue

                        elif label["label"] in ["Top of the right shoulder"]:
                            continue

                        elif label["label"] in ["Top of the shoulder"]:
                            continue

                        elif label["label"] in ["Top of the left ear","Top of left ear","Top of the left aar","Base of left ear"]:
                            continue

                        elif label["label"] in ["Top of the right ear","Top of right ear","Right ear"]:
                            continue

                        elif label["label"] in ["Bottom of the right ear","Bottom of right ear","End of the right ear","End of right ear"]:
                            continue

                        elif label["label"] in ["Bottom of the left ear","Bottom of left ear","End of the left ear","End of left ear"]:
                            continue

                        elif label["label"] in ["Widest point of the left ear","Widest point of left ear","Widest point on the left ear"]:
                            points[label_names.index("Widest point of left ear")]=point

                        elif label["label"] in ["Widest point of the right ear","Widest point of right ear","Widest point on the right ear"]:
                            points[label_names.index("Widest point of right ear")]=point

                        else:
                            print(label["label"])

                    point_list = []
                    excludes_indexes = get_exclude_indexes()
                    for n in range(len(label_names)):
                        if n in excludes_indexes:
                            continue 

                        if n in points:
                            point_list.append([points[n][0],points[n][1]])
                        else:
                            point_list.append([0.0,0.0])
                    #print(os.path.join(data_dir,folder,filename+".jpg"),[xmax,ymax,xmin,ymin])
                    if os.path.exists(os.path.join(data_dir,folder,filename+".jpg")):
                        images_list.append(os.path.join(data_dir,folder,filename+".jpg"))
                        keypoints_list.append(point_list)
                    elif os.path.exists(os.path.join(data_dir,folder,filename+".JPG")):
                        images_list.append(os.path.join(data_dir,folder,filename+".JPG"))
                        keypoints_list.append(point_list)

    return (images_list, keypoints_list)

def get_artifacts_alphapose_anno(data_dir,alphapose_test_results_file ):
    exclude_index =get_exclude_indexes()
    exclude_image_ids = ["1003_IMG_2415",
    "1007_IMG_9873","1020_IMG_5751","1022_IMG_1616",
    "1022_IMG_2034","1029_IMG_4413","1033_IMG_1161",
    "1033_IMG_1288","1034_IMG_1960","1002_IMG_9442"]

    with open(alphapose_test_results_file, errors='ignore') as json_data:
     data_dict = json.load(json_data, strict=False)
    
    DEFAULT_KEYPOINT_SCORE = 0.7
    images_list = []
    keypoints_list = []
    for each in data_dict:
        bbox = each["bbox"]
        xmin, ymin, xmax, ymax = bbox[0],bbox[1],bbox[2],bbox[3]
        image_path = each["image_id"]
        parts = image_path.split("/")

        filename, ext = os.path.splitext(parts[-1])
        if filename in exclude_image_ids:
            continue 


        image_path = os.path.join(data_dir,filename+".png")
        
        keypoints = np.array(each["keypoints"]).reshape(20,3)
        normalize_keypoints = []
        for idx, kpt in enumerate(keypoints):
            
            if idx in exclude_index:
                
                continue
            
            x, y, score = kpt[0],kpt[1],kpt[2]
            if score >= DEFAULT_KEYPOINT_SCORE:
                new_x = utils.normalize_keypoints(x,xmax,xmin)
                new_y = utils.normalize_keypoints(y,ymax,ymin)
                normalize_keypoints.append([new_x,new_y])
            else:
                normalize_keypoints.append([0.0,0.0])

        ## append to list
        images_list.append(image_path)
        print(len(normalize_keypoints))
        keypoints_list.append(normalize_keypoints)

    return (images_list, keypoints_list)
                
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_type",default="manual")
    parser.add_argument("--data_dir",default="/Volumes/My Storage/Elephant Data/DATA")
    parser.add_argument("--alpha_pose_results", default="alphapose_results/test_gt_kpt.json",required=False)
    parser.add_argument("--pickle_file",default="pickles/image_data_normalized.p")
    args = parser.parse_args()
    anno_type = args.anno_type
    data_dir = args.data_dir

    if anno_type=="manual":
        ## build pickle file - images, keypoints
        images_list, keypoints_list = get_artifacts_manual_anno(data_dir)
    elif anno_type=="alphapose":
        print("in progress")
        images_list, keypoints_list = get_artifacts_alphapose_anno(data_dir,args.alpha_pose_results)
    else:
        print("Not implemented!")
        exit()

    pickle_file = args.pickle_file
    save_annotations_pickle_file({"images":images_list,"keypoints":keypoints_list},filename=pickle_file)
    print(f" Pickle file: {pickle_file}")


