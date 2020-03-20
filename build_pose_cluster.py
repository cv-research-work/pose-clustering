#!/usr/bin/env python3

import argparse
import os
from utils import load_pickle_file
import numpy as np
import shutil
import cv2
import pdb
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
## define constants
DEFAULT_MAX_SIZE_CLUSTER = 5

## kmeans
DEFAULT_K_VALUE = 15
DEFAULT_MAX_ITERATIONS_KMEANS = 100
DEFAULT_MAX_TOLERANCE = 0.0001


## mean shift
DEFAULT_KERNEL_BANDWIDTH=0.5
DEFAULT_THRESHOLD = 0.01
def compute_distance(center,keypt):
    z1 = np.array([[(c[0],c[1]) for c in center]])
    z2 = np.array([[(c[0],c[1]) for c in keypt]])

    #distance = np.linalg.norm(z1[0],z2[0])
    distance =  sum(np.linalg.norm(x-y) for x, y in zip(z1, z2))
    return distance

def compute_distance_cosine(center, keypt):
    distance = 0
    for i in range(len(center)):
        center_x,center_y = center[i][0],center[i][1]
        keypt_x,keypt_y = keypt[i][0],keypt[i][1]
        distance += 2 * cosine([center_x,center_y],[keypt_x,keypt_y])
    
    return math.sqrt(distance)
"""
Basic clustering method, grps similar pose images into one cluster based on min distance, find large grps and update cluster centers
"""
def cluster_basic(keypoints,save_folder):
    centers = {}
    grps ={}

    centers[0] = keypoints[0]
    grps[0] = []


    for index, keypoint in enumerate(keypoints):
        add_cluster = False
        min_score = 0.8
        choosen_center = -1

        for center_index, center_keypoint in (centers.items()):
            dist = compute_distance(center_keypoint,keypoint)

            if dist < min_score:
                add_cluster = True
                min_score = dist
                choosen_center = center_index

        if add_cluster:
            grps[choosen_center].append(index)
        else:
            centers[index] = keypoint
            grps[index] = [index]

    ## check grps with large number and realign centers
    def count_large_grps(grps, max_size):
        total_size = 0
        count = 0
        
        large_grps = []
        for key, val in grps.items():
            if len(val) >= max_size:
                count += 1
                total_size += len(val)
                large_grps.append(val)
        return large_grps, count, total_size
    

    grps_with_large_cnts, number_of_grps, total_size = count_large_grps(grps, DEFAULT_MAX_SIZE_CLUSTER)
    
    ## realign center
    def update_centers(point_grps, keypoints):
        grps = {}
        # Iterating over point groups
        # to find best center for each group
        for points in point_grps:
            min_dist = 100000
            max_point = points[0] # Point which is farthest away from center for this group
            curr_center = points[0]
            for center_point in points:
                total_dist = 0
                max_dist = 0
                max_dist_point = center_point 
                for point in points:
                    curr_dist = np.linalg.norm(keypoints[center_point]-keypoints[point])
                    if curr_dist > max_dist:
                        max_dist = curr_dist
                        max_dist_point = point
                    total_dist += curr_dist
                    
                if total_dist < min_dist:
                    curr_center = center_point
                    min_dist = total_dist
                    max_point = max_dist_point
            grps[curr_center] = (points, max_point)

        return grps

    realigned_points = update_centers(grps_with_large_cnts,keypoints)
    
    ## save results:
    if os.path.exists(f"{save_folder}"):
        shutil.rmtree(f"{save_folder}", ignore_errors=False, onerror=None)
   
    os.makedirs(f"{save_folder}")

    for k, v in realigned_points.items():
        if not os.path.exists(os.path.join(f"{save_folder}",str(k))):
            os.makedirs(os.path.join(f"{save_folder}",str(k)))
        
        imgs, center = v[0],v[1]
        for index in imgs:
            img_path = images[index]
            shutil.copyfile(img_path,os.path.join(f"{save_folder}",str(k),os.path.basename(img_path)))

    return 

"""
Kmeans - clustering algorithm

Takes n keypoints as input, use default k value (specifies how many clusters are possible in the dataset), output k cluster centroids and cluster grps that maps
data points to centroid

"""
def cluster_kmeans(keypoints, k,save_folder):
    dist = np.mean
    n = len(keypoints)
    
    ## set seed:
    np.random.seed()

    ### randomly initialize k cluster centers
    clusters = keypoints[np.random.choice(n, k, replace=False)]
    new_clusters = np.zeros(shape=clusters.shape)

    previous_centers = np.zeros(shape=(n,))
    new_centers = np.zeros(shape=(n,))
    isOptimal = True


    for i in range(DEFAULT_MAX_ITERATIONS_KMEANS):        
        for index in range(n):
            distances = [np.linalg.norm(keypoints[index] - clusters[centroid]) for centroid in range(k)]
            classification = distances.index(min(distances))
            new_centers[index] = classification
        
        ## update cluster centers
        for cluster in range(k):
            new_clusters[cluster] = dist(keypoints[new_centers==cluster], axis=0)
            original = clusters[cluster]
            curr = new_clusters[cluster]

            if not np.all(original==0):
                if np.sum((curr - original)/original * 100.0) > DEFAULT_MAX_TOLERANCE:
                    isOptimal = False

        if isOptimal:
            break

        previous_centers = new_centers
        clusters = new_clusters


    #print(f"No of iterations: {i}")
    ## save cluster images
    ## display clusters
    plt.rcParams['figure.figsize'] = (16,9)
    plt.style.use('ggplot')

    colors = 10*["r", "g", "c", "b", "k"]
    
    # Perform PCA analysis
    #mean = np.empty((0))
    #mean, eigenvectors, eigenvalues = cv2.PCACompute2(keypoints, mean)
    #plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c=nearest_clusters,s=50, cmap='viridis');
    #plt.show()

    if os.path.exists(f"{save_folder}"):
        shutil.rmtree(f"{save_folder}", ignore_errors=False, onerror=None)
   
    os.makedirs(f"{save_folder}")


    for k in range(DEFAULT_K_VALUE):
        
        indexs = [i for i, o in enumerate(new_centers) if o==k]
        
        ## save results:
        if not os.path.exists(f"{save_folder}/{str(k)}"):
            os.makedirs(f"{save_folder}/{str(k)}")

        
        for v in indexs:
            v = int(v)
            img_path = images[v]
            shutil.copyfile(img_path,os.path.join(f"{save_folder}",str(k),os.path.basename(img_path)))

    return new_centers, clusters

def compute_sse(k, data,clustering, centers):
    n = data.shape[0]
    sse = 0
    for i in range(n):
        for kk in range(k):
            c_id = clustering[i]
            if c_id == kk:
                sse += np.linalg.norm(centers[kk]-data[i])

    print("%4f"%(sse))
    return sse

"""
Mean shift  = shift points and cluster
"""
def mean_shift(keypoints,kernel_bandwidth=DEFAULT_KERNEL_BANDWIDTH):
    guassian_kernel = lambda distance, band_width:  (1 / (band_width * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / band_width)) ** 2)
    kernel = guassian_kernel

    def shift(point, all_keypoints, kernel_bandwidth):
        shift_xy = {}
        weights = {}

        for i, kpt in enumerate(keypoints):
            distance = compute_distance(point,kpt)
            weight = guassian_kernel(distance,kernel_bandwidth)

            for o in range(len(kpt)):
                x , y = kpt[o][0],kpt[o][1]
                if o not in shift_xy:
                    shift_xy[o] = []
                    shift_xy[o].append((x * weight))
                    shift_xy[o].append((y * weight))
                else:
                    shift_xy[o][0] = shift_xy[o][0] + (x * weight)
                    shift_xy[o][1] = shift_xy[o][1] + (y * weight)
                
                if o not in weights:
                    weights[o] =  0.0
               
                weights[o] = weights[o] + weight

                
        
        ## compute avg shift
        new_point = np.array([[shift_xy[i][0]/weights[i],shift_xy[i][1]/weights[i]] for i in range(len(weights))])
        
        return np.array(new_point)
            
    

    #shift points
    shift_keypoints = np.array(keypoints)
    n = shift_keypoints.shape[0]

    ##
    shifting = [True] * keypoints.shape[0]
    
    ## mean shift points
    while True:
        max_distance = 0
        for i in range(n):
            if not shifting[i]:
                continue
            ## point is shifted, 
            p_shift_init = shift_keypoints[i].copy()
            shift_keypoints[i] = shift(shift_keypoints[i],keypoints,kernel_bandwidth)

            ## compute distance between original point and shifted point
            dist = compute_distance(shift_keypoints[i], p_shift_init)
            max_distance = max(max_distance,dist)

            ## shift or not shift
            shifting[i] = dist > DEFAULT_THRESHOLD

        print(max_distance)
        if max_distance < DEFAULT_THRESHOLD:
            break

    return shift_keypoints


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file",default="pickles/image_data_normalized.p")
    parser.add_argument("--cluster",default="basic")
    parser.add_argument("--save_folder",default="save_results")

    args = parser.parse_args()
    pickle_file = args.pickle_file
    cluster = args.cluster
    images, keypoints  = load_pickle_file(args.pickle_file)
    keypoints = np.array(keypoints)
    
    keypoints = keypoints.reshape(keypoints.shape[0],20,2)
    
    if cluster == "basic":
        save_path = args.save_folder
        cluster_basic(keypoints,save_path)

    elif cluster == "kmeans":
        grps, centers = cluster_kmeans(keypoints,DEFAULT_K_VALUE,args.save_folder)
        #sse = compute_sse(k, keypoints,grps, centers)
    elif cluster == "mean_shift":
        save_path = args.save_folder
        new_points = mean_shift(keypoints)
        cluster_basic(new_points,save_path)
    else:
        print(f"Clustering algorithm not implemented")
        exit()

    
    

