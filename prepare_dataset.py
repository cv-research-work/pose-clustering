#!/usr/bin/env python3
import sys
import os
import pandas as pd
import shutil
import json
import numpy as np
import pickle
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
dataset_path = "data"
def getImagesAnnotations():
    df = {}

    images = []
    keypoints = []
    labels = []
    
    label_names = ["Top of the head",'Highest point on the back',
                "Left eye","Right eye","Jaw",
                #"Eyebrow of the face",
                "Base of trunk",
                #"End of trunk",
                "Left elbow",
                "Right elbow",
                "Bottom of the belly","Left knee","Right knee",
                #"Top of left ear","Top of right ear",
                
                "Widest point of left ear",
                "Widest point of right ear",
                #"Bottom of left ear","Bottom of right ear",
                #"Top of the left shoulder","Top of the right shoulder",
                #"Top of the shoulder",
                
                "Base of tail",
                #"End of tail", 
                "Bottom of the right backfoot",
                "Bottom of the right front foot",
                "Bottom of the left front foot",
                "Bottom of the left backfoot",
                "Base of right tusk",
                #"End of right tusk",
                "Base of left tusk",
                #"End of left tusk"
                ]
    
    ## Rename and copy images to new folder..
    ## Loop through dataset path
    img_id = 1
    scale_factor = 8
    save_path = "elephant_images"

    allowed_width = 256
    allowed_height = 192

    ## xmin, xmax, ymin, ymax

    dataset_path ="data"
    for folder in os.listdir(dataset_path):
        if folder not in [".DS_Store"]:
            index = 0
            for f in os.listdir(os.path.join(dataset_path,folder)):
                if f==".DS_Store":
                    continue
                
                filename, ext = os.path.splitext(f)
                if ext==".MOV":
                    continue

                if not os.path.exists(os.path.join(dataset_path,folder,filename+".json")):
                    print(folder,f)
                    continue
                
                if ext in [".JPG",".jpg"]:
                    ## rename and copy
                    ## resize
                    original_img = cv2.imread(os.path.join(dataset_path,folder, f)) 
                    dimensions = original_img.shape
                    #height, width, channels = img.shape
                    resized_image=cv2.resize(original_img,(allowed_width,allowed_height)) 

                    #Save the result on disk in the "elephant_images" folder
                    if not os.path.exists(os.path.join(save_path,folder)):
                        os.makedirs(os.path.join(save_path,folder))

                    cv2.imwrite(os.path.join(save_path,folder,f),resized_image)
                    images.append(os.path.join(save_path,folder,f))
                    
                elif ext in [".json"]:
                    points = {}
                    if folder=="awa":
                        img = cv2.imread(os.path.join(save_path,folder,filename+".jpg"))
                    else:
                        img = cv2.imread(os.path.join(save_path,folder,filename+".JPG"))
                    height, width, channels = img.shape
                    
                    ## Dictionary of Elephant Image
                    ele_img = {}
                    ele_img[u'license'] = 0
                    ele_img[u'file_name'] = filename+".JPG"
                    ele_img[u'width'] = width
                    ele_img[u'height'] = height

                    ## bounding box coordinates
                    xml_path = os.path.join(dataset_path,folder,filename+".xml")
                    if os.path.exists(xml_path):
                        xml_file = xml_path
                        print(xml_file)
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
                                            print(i)
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
                    with open(os.path.join(dataset_path,folder,f)) as json_file:
                        data = json.load(json_file)
                        shapes = data["shapes"]

                        points = {}

                        scale_factor_x = dimensions[1]/allowed_width
                        scale_factor_y = dimensions[0]/allowed_height

                        x1p = xmin / dimensions[1]
                        x2p = xmax / dimensions[1]
                        y1p = ymin / dimensions[0]
                        y2p = ymax / dimensions[0]

                        
                        pts_list = []
                        xmin = x1p*allowed_width
                        ymin = y1p*allowed_height
                        xmax = x2p*allowed_width
                        ymax = y2p*allowed_height
                        print(xmin,ymin,xmax,ymax)
                       
                        for label in shapes:

                            ## keep track of available labels
                            if label["label"] not in labels:
                                labels.append(label["label"])
                            
                            x, y =   label["points"][0][0] , label["points"][0][1]
                            
                            x = x/scale_factor_x
                            y = y/scale_factor_y

                            pts_list.append([x,y])

                            ### in COCO,v = 0,not labeled;v = 1;labeled but invisible;v = 2,labeled and visible
                            point = [x,y,2]
                            if label["label"] in ['Top of the head','Highest point of the head']:
                                points[label_names.index('Top of the head')]=point

                            elif label["label"] in ['Highest point on the back', 
                            'Highest point of the back','Highest point on back']:
                                points[label_names.index('Highest point on the back')] = point

                            elif label["label"] in ["Left eye","Left ey"]:
                                points[label_names.index('Left eye')] = point
                            elif label["label"] in ["Right eye"]:
                                points[label_names.index('Right eye')] = point
                            elif label["label"] in ["Jaw"]:
                                points[label_names.index("Jaw")] = point

                            elif label["label"] in ["Eyebrow of the face"]:
                                #points[label_names.index("Eyebrow of the face")] = point
                                continue

                            elif label["label"] in ["Base of trunk","Base of the trunk"]:
                                points[label_names.index("Base of trunk")] = point
                            elif label["label"] in ["End of trunk","End of the trunk"]:
                                #points[label_names.index("End of trunk")] = point
                                continue
                            elif label["label"] in ["Base of tail","Base of the tail","Base of tal"]:
                                points[label_names.index("Base of tail")] = point
                            elif label["label"] in ["End of tail","End of the tail"]:
                                #points[label_names.index("End of tail")] = point
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

                            elif label["label"] in ["Base of left tusk","Base of the left tusk"]:
                                points[label_names.index("Base of left tusk")] = point

                            elif label["label"] in ["Base of right tusk","Base of the right tusk"]:
                                points[label_names.index("Base of right tusk")] = point

                            elif label["label"] in ["End of right tusk","End of the right tusk"]:
                                #points[label_names.index("End of right tusk")] = point
                                continue

                            elif label["label"] in ["End of left tusk","End of the left tusk"]:
                                #points[label_names.index("End of left tusk")] = point
                                continue

                            elif label["label"] in ["Bottom of the right backfoot",
                            "Bottom of right backfoot","Right backfoot","Bottom of the right back foot"]:
                                points[label_names.index("Bottom of the right backfoot")] = point

                            elif label["label"] in ["Bottom of the left backfoot",
                            "Bottom of left backfoot","Left backfoot","Bottom of the left back foot"]:
                                points[label_names.index("Bottom of the left backfoot")] = point

                            elif label["label"] in ["Bottom of the left front foot",
                            "Bottom of left front foot"]:
                                points[label_names.index("Bottom of the left front foot")] = point

                            elif label["label"] in ["Bottom of the right front foot",
                            "Bottom of right front foot"]:
                                points[label_names.index("Bottom of the right front foot")] = point

                            elif label["label"] in ["Top of the left shoulder"]:
                                #points[label_names.index("Top of the left shoulder")] = point
                                continue

                            elif label["label"] in ["Top of the right shoulder"]:
                                #points[label_names.index("Top of the right shoulder")] = point
                                continue

                            elif label["label"] in ["Top of the shoulder"]:
                                #points[label_names.index("Top of the shoulder")] = point
                                continue


                            elif label["label"] in ["Top of the left ear","Top of left ear","Top of the left aar","Base of left ear"]:
                                #points[label_names.index("Top of left ear")] = point
                                continue

                            elif label["label"] in ["Top of the right ear","Top of right ear","Right ear"]:
                                #points[label_names.index("Top of right ear")]=point
                                continue

                            elif label["label"] in ["Bottom of the right ear","Bottom of right ear","End of the right ear","End of right ear"]:
                                #points[label_names.index("Bottom of right ear")]=point
                                continue

                            elif label["label"] in ["Bottom of the left ear","Bottom of left ear","End of the left ear","End of left ear"]:
                                #points[label_names.index("Bottom of left ear")]=point
                                continue

            
                            elif label["label"] in ["Widest point of the left ear","Widest point of left ear","Widest point on the left ear"]:
                                points[label_names.index("Widest point of left ear")]=point


                            elif label["label"] in ["Widest point of the right ear","Widest point of right ear","Widest point on the right ear"]:
                                points[label_names.index("Widest point of right ear")]=point

                            else:
                                print(label["label"])

                        kptCoco = []
                        num_keypoints = 0

                        for n in range(len(label_names)):
                            if len(points)==0:
                                continue
                            if n in points:
                                cx, cy, score = points[n]
                                visibility = 1
                                num_keypoints+=1
                            else:
                                cx, cy, score = 0,0,0
                                visibility = 0
                            
                            kptCoco.append([[cx,   visibility],
                                            [cy,  visibility],
                                            [  0.,   0.]])
                        
                        #xmin, ymin, , ymax = bbox
                        ele_img['bbox']=[xmin, ymin, xmax, ymax]
                        ele_img['joints_3d'] = np.asarray(kptCoco)
                        ele_img['num_keypoints'] = num_keypoints
                        print(ele_img['bbox'])
                        keypoints.append(ele_img)
                        
    return images, keypoints

def split_train_test(images,keypoints):
    n_train = len(images)
    print(f"Total no of images:{n_train}")
    indices = np.arange( n_train )
    np.random.permutation ( indices )
    split_percent = 0.1
    train_split = (1 - split_percent)
    test_split = split_percent

    train_sample = indices[:  int ( n_train * train_split )]
    test_sample = indices[:  int ( n_train * test_split )]

    x_train, y_train, x_test, y_test = [images[i] for i in train_sample],[keypoints[i] for i in train_sample], [images[i] for i in test_sample], [keypoints[i] for i in test_sample]
    print(f"Total no of images training:{len(x_train)}, test images: {len(x_test)}")

    return x_train, y_train, x_test, y_test

def save_annotations_pickle_file(train_dict, filename=""):
    f = open(filename,"wb")
    pickle.dump(train_dict,f,-1)
    f.close()    
    
def load_annotations(filename):
    with open(filename, 'rb') as fid:
        items, labels = pickle.load(fid)
    return (items,labels)

def copy_test_images_folder(x_test,y_test):
    test_path = "test_images"
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
    
    for i in range(len(y_test)):
        file = x_test[i]
        #file = file.replace("elephant_images","data")
        print(file)
        shutil.copyfile(file, os.path.join(test_path,y_test[i]["file_name"]))
        print(f"Copied file:{file}")
    return 

        
images, keypoints = getImagesAnnotations()
x_train, y_train, x_test, y_test = split_train_test(images,keypoints)

## train folder - test folder - move images
#  Build pkl file
## train pkl file
print(len(x_train))

##  train
train_file = "train_annot_keypoint.pkl"
save_annotations_pickle_file((x_train,y_train), train_file)
(items, labels) = load_annotations(train_file)
test_file = "test_annot_keypoint.pkl"
save_annotations_pickle_file((x_test,y_test), test_file)
exit()
## Test Images - copy to test folder
copy_test_images_folder(x_test,y_test)

## test

#xmin, ymin, , ymax = bbox
test_list = []

for i in range(len(y_test)):
    x = {}
    x["bbox"]= y_test[i]["bbox"]
    x["joints_3d"] = np.asarray([[[2,   2],
                                [3,  2],
                                [  0.,   0.]]])
    x["num_keypoints"]= None
    x[u'license'] = y_test[i]["license"]
    x[u'file_name'] = y_test[i]["file_name"]
    x[u'width'] = y_test[i]["width"]
    x[u'height'] = y_test[i]["height"]
    test_list.append(x)

save_annotations_pickle_file((x_test,test_list), test_file)
(titems, tlabels) = load_annotations(test_file)
print(tlabels)


