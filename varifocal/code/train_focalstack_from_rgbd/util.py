# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import tensorflow as tf
import math, cv2
from PIL import Image
import matplotlib.pyplot as plt


def np_space_to_depth(x, block_size):
    x = np.asarray(x)
    batch, height, width, depth = x.shape #NHWC
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(batch, reduced_height, block_size,
                         reduced_width, block_size, depth)
    z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
    return z

# interleave numpy array
def interleave_np(r,x):
    #NCHW
    if (r==1):
        return x
    #elif (r==2):
    #    H,W = x.shape[2], x.shape[3]   
    #    return np.concatenate((x[:,:,0:H:r,0:W:r],x[:,:,0:H:r,1:W:r],x[:,:,1:H:r,0:W:r],x[:,:,1:H:r,1:W:r]),axis=1) 
    else:        
        return np.transpose(np_space_to_depth(np.transpose(x, (0,2,3,1)), r), (0,3,1,2))

# interleave tensor
def interleave(r,x):
    #NCHW
    if (r==1):
        return x
    else:
        return tf.space_to_depth(x, r, data_format='NCHW')

# deinterleave tensor
def deinterleave(r,x):
    if (r==1):
        return x
    else:
        return tf.depth_to_space(x,r,data_format='NCHW')

def init_weights(shape, init_method='xavier', xavier_params = (None, None), r = 0.5):
    #xavier
    (fan_in, fan_out) = xavier_params        
    filtersize = shape[1]*shape[2]
    high = np.sqrt(r*2.0/(fan_in+fan_out))
    low = -high
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def arcsin_hack(x):
    x = np.minimum(x, 1)
    x = np.maximum(x, -1)
    return np.arcsin(x)

def circ_filter(w): # reimplement from matlab function fspecial('disk', rad)
    rad = w/2
    rad = np.maximum(rad, 1e-6) # prevent zero-divide
    rad2 = rad**2
    crad  = np.ceil(rad-0.5) 
    l = np.linspace(-crad,crad,2*crad+1)
    [x,y] = np.meshgrid(l,l)
    maxxy = np.maximum(np.abs(x), np.abs(y))
    minxy = np.minimum(np.abs(x), np.abs(y))
    m1 = (rad2 < (maxxy+0.5)**2 + (minxy-0.5)**2)*(minxy-0.5) + (rad2 >= (maxxy+0.5)**2 + (minxy-0.5)**2)*np.sqrt(np.abs(rad2 - (maxxy + 0.5)**2))
    m2 = (rad2 > (maxxy-0.5)**2 + (minxy+0.5)**2)*(minxy+0.5) + (rad2 <= (maxxy-0.5)**2 + (minxy+0.5)**2)*np.sqrt(np.abs(rad2 - (maxxy - 0.5)**2))
    sgrid = (rad2*(0.5*(arcsin_hack(m2/rad) - arcsin_hack(m1/rad)) + 0.25*(np.sin(2*arcsin_hack(m2/rad)) - np.sin(2*arcsin_hack(m1/rad)))) - 
            (maxxy-0.5)*(m2-m1) + (m1-minxy+0.5))*((np.logical_or(np.logical_and((rad2 < (maxxy+0.5)**2 + (minxy+0.5)**2) , (rad2 > (maxxy-0.5)**2 + (minxy-0.5)**2)),
	        (np.logical_and(np.logical_and((minxy==0),(maxxy-0.5 < rad)),(maxxy+0.5>=rad))))))
    sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < rad2)
    cradint = np.int(crad)
    sgrid[cradint,cradint] = np.minimum(np.pi*rad2,np.pi/2)
    if ((crad>0) and (rad > crad-0.5) and (rad2 < (crad-0.5)**2+0.25)): 
        m1  = np.sqrt(rad2 - (crad - 0.5)**2)
        m1n = m1/rad
        sg0 = 2*(rad2*(0.5*arcsin_hack(m1n) + 0.25*np.sin(2*arcsin_hack(m1n)))-m1*(crad-0.5))
        sgrid[2*cradint,cradint] = sg0
        sgrid[cradint,2*cradint] = sg0
        sgrid[cradint,0]        = sg0
        sgrid[0,cradint]        = sg0
        sgrid[2*cradint-1,cradint]   = sgrid[2*cradint-1,cradint] - sg0
        sgrid[cradint,2*cradint-1]   = sgrid[cradint,2*cradint-1] - sg0
        sgrid[cradint,1]        = sgrid[cradint,1]      - sg0
        sgrid[1,cradint]        = sgrid[1,cradint]      - sg0

    sgrid[cradint,cradint] = np.minimum(sgrid[cradint,cradint],1)
    kernel = (sgrid/np.sum(sgrid)).astype(dtype=np.float32)

    return kernel.shape[0], kernel


def calCoC(D, film_len_dist, dp_focal, fov, H, W, imW, dm, cocScale):
    dp_dm = 1./dm.reshape((H,W)) # diopter
    dp_focal_length = 1./film_len_dist + dp_focal
    coc_physical = D * abs((dp_focal_length - dp_dm)/(dp_focal_length - dp_focal) - 1)
    film_width = 2*film_len_dist*np.tan(fov/2.)
    coc_pixels = coc_physical/film_width*imW
    coc_pixels = coc_pixels / cocScale  
    return coc_pixels

def calImageGradients(images):
    # x is a 4-D tensor NCHW
    dx = images[:, :, :, 1:] - images[:, :, :, :-1]
    dy = images[:, :, 1:, :] - images[:, :, :-1, :]
    return dx, dy


def cal_ssim_focalstack(recon_fs, true_fs, crop_width):
    #C-H-W-N
    C = recon_fs.shape[0]
    H = recon_fs.shape[1]
    W = recon_fs.shape[2]
    N = recon_fs.shape[3]
    SSIMs = np.zeros((N))
    y_pred = tf.transpose(recon_fs[:,crop_width:H-crop_width,crop_width:W-crop_width,:], [3,1,2,0]) #NHWC
    y_true = tf.transpose(true_fs[:,crop_width:H-crop_width,crop_width:W-crop_width,:], [3,1,2,0]) #NHWC
    SSIMs = tf.image.ssim(y_true, y_pred, max_val = 1.0)
    return SSIMs


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def cal_psnr_focalstack(recon_fs, true_fs, crop_width):
    C = recon_fs.shape[0]
    H = recon_fs.shape[1]
    W = recon_fs.shape[2]
    N = recon_fs.shape[3]
    PSNRs = np.zeros((N))
    for n in range(N):
        diff = recon_fs[:,crop_width:H-crop_width,crop_width:W-crop_width,n] - true_fs[:,crop_width:H-crop_width,crop_width:W-crop_width,n]
        rmse = math.sqrt(np.mean(diff**2.))
        PSNRs[n] = 20*math.log10(1.0/rmse)
    return PSNRs


def save_png(x, fn):
    im = Image.fromarray((255*x).astype('uint8'))                    
    im.convert('RGB').save(fn)  

def save_tiff(x, fn):
    im = (65535*x).astype('uint16')     
    im = np.flip(im, axis=2)
    cv2.imwrite(fn, im)