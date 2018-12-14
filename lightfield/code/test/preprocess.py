# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from util import *

def normalize_rgb_exr(im, GLOBAL_SCALE):
    im = np.power(abs(np.asarray(im)), 1/2.2) # b-g-r
    im = np.flip(im, axis=2) # r-g-b
    im[np.isnan(im)] = 0
    im_max = GLOBAL_SCALE #np.amax(im)
    im_min = 0 
    im = np.minimum(im, GLOBAL_SCALE)
    im = (im - im_min)/(im_max-im_min) # normalize
    return im, im_max, im_min        


def rescale_depth_exr(im, depthScale):
    im = im/depthScale 
    return im


def loadData(data_path, H, W, imH, imW, numViewsYX, validSceneIdxList, numChannels,depthScale, diopterScale, LOAD_RGBD, GLOBAL_SCALE):
    print("load data...")

    numScenes = validSceneIdxList.size    
    print("found %d valid scenes" % numScenes)
    if numScenes==0:
       input("error: please correct data path!")

    patch_shift = np.int(np.floor(W/2)) #W/2 # when patch_shift<W, overlapped patches are allowed
    num_patch_per_row = np.int(np.floor((imW-W)/patch_shift+1))
    num_patch_per_col = np.int(np.floor((imH-H)/patch_shift+1))
    num_patch_per_img = num_patch_per_row*num_patch_per_col
    numSamples = numScenes*num_patch_per_img*numChannels # number of sample patches
  
    numViews = numViewsYX[0]*numViewsYX[1]
    cleanRGB = np.zeros((numSamples, H, W, numViews),dtype=np.float32)
    diopter = np.zeros((numSamples, H, W, numViews),dtype=np.float32)
  
    for s in range(0, numScenes): # 1:numScenes
       v = validSceneIdxList[s]       
       print("load scene %03d" % v)
       
       # load RGBD data
       if LOAD_RGBD:
           for viewY in range(0, numViewsYX[0]): #
            for viewX in range(0, numViewsYX[1]): #

               view = viewY*numViewsYX[1]+viewX
               global_frame_index = v*numViews+view
               
               # load sharp RGB images
               im = cv2.imread("%s/frame%05d.exr" % (data_path,global_frame_index), -1)        
               im, im_max, im_min = normalize_rgb_exr(im,GLOBAL_SCALE)

               for row in range(0, num_patch_per_row):
                 for col in range(0, num_patch_per_col):
                       patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :numChannels]
                       idx_patch = row*num_patch_per_col + col
                       for c in range(0, numChannels):
                           idx = s*num_patch_per_img*numChannels+idx_patch*numChannels+c
                           cleanRGB[idx,:,:, view] = patch[:,:,c]

               # load depth map
               im = cv2.imread("%s/frame_depth%05d.exr" % (data_path, global_frame_index), -1) 
               im = rescale_depth_exr(im, depthScale) # use global scale (not per scene)
               for row in range(0, num_patch_per_row):
                 for col in range(0, num_patch_per_col):
                    patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :]
                    idx_patch = row*num_patch_per_col + col
                    for c in range(0, numChannels): 
                        idx = s*num_patch_per_img*numChannels+idx_patch*numChannels+c
                        depth = (patch[:,:,0]).reshape((1,H,W))
                        diopter[idx,:,:, view] = 1/(depthScale*depth)/diopterScale
                                      


    print("data imported")
    return numSamples, cleanRGB, diopter

 
