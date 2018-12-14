# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

def normalize_rgb_exr(im, GLOBAL_SCALE):
    im = np.power(abs(np.asarray(im)), 1/2.2) # b-g-r
    im = np.flip(im, axis=2) # r-g-b
    im[np.isnan(im)] = 0
    im_max = GLOBAL_SCALE 
    im_min = 0
    im = np.minimum(im, GLOBAL_SCALE) 
    im = (im - im_min)/(im_max-im_min) # normalize
    return im, im_max, im_min      

def rescale_rgb_exr(im, im_max, im_min,GLOBAL_SCALE):
    im = np.power(abs(np.asarray(im)), 1/2.2) # b-g-r
    im = np.flip(im, axis=2) # r-g-b
    im[np.isnan(im)] = 0
    im = np.minimum(im, GLOBAL_SCALE) 
    im = (im - im_min)/(im_max-im_min)
    return im

def rescale_depth_exr(im, depthScale):
    im = im/depthScale
    return im

def loadData(data_path, RENORM_SCALE, N, M, H, W, imH, imW, dp_focal,dp_display,fov,D,startSceneIDX,endSceneIDX,numEvalScenes,numChannels, depthScale, diopterScale, LOAD_FS,LOAD_RGBD,GLOBAL_SCALE):
    print("preprocessing data...")

    numScenes = 0
    validSceneIdxList = []
    for s in range(startSceneIDX, endSceneIDX+1):
        if os.path.exists("%s" % (data_path)):
           validSceneIdxList.append(s)
           numScenes = numScenes+1

    print("found %d valid scenes" % numScenes)
    if numScenes==0:
       input("error: please correct data path!")

    patch_shift = np.int(np.floor(W/2)) #W/2 # when patch_shift<W, overlapped patches are allowed
    num_patch_per_row = np.int(np.floor((imW-W)/patch_shift+1))
    num_patch_per_col = np.int(np.floor((imH-H)/patch_shift+1))
    num_patch_per_img = num_patch_per_row*num_patch_per_col
    numSamples = numScenes*num_patch_per_img # number of sample patches

    numTrainSamples = (numScenes-numEvalScenes)*num_patch_per_img # number of samples for training
    numEvalSamples = numSamples - numTrainSamples # number of samples for validation during training
    
    focal_stack = np.zeros((numSamples, H, W, numChannels, N),dtype=np.float32) 
    cleanRGB = np.zeros((numSamples, H, W, numChannels),dtype=np.float32) 
    depth = np.zeros((numSamples, H, W),dtype=np.float32) 
    diopter = np.zeros((numSamples, H, W),dtype=np.float32) 
 
    # assume all display planes perfectly cover the field of view 
    pixel_sizes = 2*np.tan(fov/2)/dp_display/imW 
    kernel_width = np.zeros((N,M))
    
    for s in range(0, numScenes): 
       
       v = validSceneIdxList[s]
       print("load scene %03d" % v)
 
       # load RGBD data
       if LOAD_RGBD:
           # load sharp RGB images
           im = cv2.imread("%s/anim02_frame_%04d_rgb.exr" % (data_path, v), -1)            
           im, im_max, im_min = normalize_rgb_exr(im,GLOBAL_SCALE)

           for row in range(0, num_patch_per_row):
             for col in range(0, num_patch_per_col):
                   patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :numChannels]
                   idx_patch = row*num_patch_per_col + col
                   idx = s*num_patch_per_img+idx_patch
                   cleanRGB[idx,:,:,:] = patch

           # load depth map
           im = cv2.imread("%s/anim02_frame_%04d_depth.exr" % (data_path, v), -1)
           im = rescale_depth_exr(im, depthScale) # use global scale (not per scene)
           for row in range(0, num_patch_per_row):
             for col in range(0, num_patch_per_col):
                patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :]
                idx_patch = row*num_patch_per_col + col
                idx = s*num_patch_per_img+idx_patch
                depth[idx,:,:] = patch[:,:,0]
                diopter[idx,:,:] = 1/(depthScale*depth[idx,:,:])/diopterScale
                   
       # load focal stack images
       if LOAD_FS:
          for i in range(0, len(dp_focal)):
             im = cv2.imread("%s/anim02_frame_%04d_focal%02d_accum.exr" % (data_path, v, i), -1)             
             im = rescale_rgb_exr(im, im_max, im_min,GLOBAL_SCALE)
             for row in range(0, num_patch_per_row):
                for col in range(0, num_patch_per_col):
                   patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :numChannels]
                   idx_patch = row*num_patch_per_col + col
                   idx = s*num_patch_per_img+idx_patch
                   focal_stack[idx,:,:,:,i] = patch
               

    # compute psf width
    for i in range(0, N):
        for j in range(0, M):             
            coc = np.abs(dp_focal[i]/dp_display[j]-1) * D / pixel_sizes[j]
            kernel_width[i,j] = coc
    print("pixels of different display planes have different size")
    print("kernel widths:")
    print(kernel_width)
                          
                        
    print("data imported")
    return focal_stack, numScenes, numTrainSamples, numEvalSamples, cleanRGB, depth, diopter, kernel_width, validSceneIdxList

