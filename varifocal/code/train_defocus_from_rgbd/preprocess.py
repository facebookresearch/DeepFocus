import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2, os

def normalize_rgb_exr(im):
    im = np.power(abs(np.asarray(im)), 1/2.2) # b-g-r
    im = np.flip(im, axis=2) # r-g-b
    im_max = np.amax(im)
    im_min = 0
    im = (im - im_min)/(im_max-im_min) # normalize
    return im, im_max, im_min        

def rescale_rgb_exr(im, im_max, im_min):
    im = np.power(abs(np.asarray(im)), 1/2.2) # b-g-r
    im = np.flip(im, axis=2) # r-g-b
    im = (im - im_min)/(im_max-im_min)
    return im

def rescale_depth_exr(im, depthScale):
    im = im/depthScale
    return im

def loadData(data_path, N, H, W, imH, imW,dp_focal,fov, D, startSceneIDX,endSceneIDX,numEvalScenes,numChannels, depthScale, diopterScale, LOAD_FS, LOAD_RGBD):
    print("preprocessing data...")

    numScenes = 0
    validSceneIdxList = []
    for s in range(startSceneIDX, endSceneIDX+1):
        if os.path.exists("%s/seed%04d/" % (data_path, s)):
           validSceneIdxList.append(s)
           numScenes = numScenes+1

    print("found %d valid scenes" % numScenes)
    if numScenes==0:
       input("error: please correct data path!")

    patch_shift = np.int(np.floor(W/2)) 
    num_patch_per_row = np.int(np.floor((imW-W)/patch_shift+1))
    num_patch_per_col = np.int(np.floor((imH-H)/patch_shift+1))
    num_patch_per_img = num_patch_per_row*num_patch_per_col
    numSamples = numScenes*num_patch_per_img*numChannels # number of sample patches

    numTrainSamples = (numScenes-numEvalScenes)*num_patch_per_img*numChannels # number of samples for training
    numEvalSamples = numSamples - numTrainSamples # number of samples for validation during training
    
    focal_stack = np.zeros((numSamples, H, W, N), dtype=np.float32) 
    cleanRGB = np.zeros((numSamples, H, W), dtype=np.float32)
    depth = np.zeros((numSamples, H, W), dtype=np.float32)
    diopter = np.zeros((numSamples, H, W), dtype=np.float32)
        
    for s in range(0, numScenes):
       
       v = validSceneIdxList[s]
       print("load scene %03d" % v)

       # load RGBD data
       if LOAD_RGBD:
           # load infocus RGB image
           im = cv2.imread("%s/seed%04d/%d/clean_pass_rgb.exr" % (data_path, v, imW), -1)    
           im, im_max, im_min = normalize_rgb_exr(im)

           for row in range(0, num_patch_per_row):
             for col in range(0, num_patch_per_col):
                   patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :numChannels]
                   idx_patch = row*num_patch_per_col + col
                   for c in range(0, numChannels):
                       idx = s*num_patch_per_img*numChannels+idx_patch*numChannels+c
                       cleanRGB[idx,:,:] = patch[:,:,c]

           # load depth map
           im = cv2.imread("%s/seed%04d/%d/clean_pass_depth_rgb.exr" % (data_path, v, imW), -1)
           im = rescale_depth_exr(im, depthScale) # use global scale (not per scene)
           for row in range(0, num_patch_per_row):
             for col in range(0, num_patch_per_col):
                patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :]
                idx_patch = row*num_patch_per_col + col
                for c in range(0, numChannels): #each color channel is used as a separate sample, and each sample needs a depth map
                    idx = s*num_patch_per_img*numChannels+idx_patch*numChannels+c
                    depth[idx,:,:] = patch[:,:,0]
                    diopter[idx,:,:] = 1/(depthScale*depth[idx,:,:])/diopterScale
                   
       # load focal stack images
       if LOAD_FS:
          for i in range(0, len(dp_focal)):
             im = cv2.imread("%s/seed%04d/%d/frame%04d.exr" % (data_path, v, imW, i), -1)             
             im = rescale_rgb_exr(im, im_max, im_min)
             for row in range(0, num_patch_per_row):
                for col in range(0, num_patch_per_col):
                   patch = im[row*patch_shift:row*patch_shift+H, col*patch_shift:col*patch_shift+W, :numChannels]
                   idx_patch = row*num_patch_per_col + col
                   for c in range(0, numChannels):
                       idx = s*num_patch_per_img*numChannels+idx_patch*numChannels+c
                       focal_stack[idx,:,:,i] = patch[:,:,c]

    print("data imported")
    return focal_stack, dp_focal, numScenes, numTrainSamples, numEvalSamples, cleanRGB, depth, diopter,validSceneIdxList
         
