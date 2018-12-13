# DeepFocus source code
# Non-Commercial use only
# Lei Xiao (lei.xiao@fb.com)
# If you use the code, please cite our work "Xiao et al., DeepFocus: Learned Image Synthesis for Computational Displays, ACM SIGGRAPH Asia 2018"

# Mode: lightfield, interpolation from sparse views, process each color channel separately


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
from util import *
from preprocess import *


# select GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# session mode selection
#MODE = "TRAIN"
MODE = "TEST"

# set true if you wan to load our pretrained model
#RESTORE_TRAINED_MODEL = False
RESTORE_TRAINED_MODEL = True 

# select data
DATA_PATH = '../data/'

# number of input views
numInputViews = 9
#numInputViews = 5

if RESTORE_TRAINED_MODEL:
   if numInputViews == 9:
       RESTORE_LOGS_DIR = "../model/saved_models-llf4d-9inputs/select/" 
   elif numInputViews ==5:
       RESTORE_LOGS_DIR = "../model/saved_models-llf4d-5inputs/select/"
   else:
       input("No trained model is available for the selected input views")


if MODE=="TRAIN":
   startSceneIDX = 0 # index of the starting scene
   endSceneIDX = 65 # index of the last scene
   numTrainScenesPerIter = endSceneIDX - startSceneIDX # number of training scenes loaded at each iteration (reduce it if can't fit all data once) 
   startEvalSceneIDX = 66 #use as validation dataset
   endEvalSceneIDX = 75 #
else:
   startSceneIDX = 76 # index of the starting scene for test
   endSceneIDX = 85 # index of the last scene

# save data directories
VERSION = "llf4d-new"
TF_SUMMARY_FLAG = "v-%s" % VERSION
if not RESTORE_TRAINED_MODEL:
   RESTORE_LOGS_DIR = "./saved_models-%s/select/" % VERSION     
LOGS_DIR = "./saved_models-%s/" % VERSION
IMGS_DIR = "./results-%s/" % VERSION
PSNR_FN = "./results-%s/test_PSNRs-%s.txt" % (VERSION, VERSION)
SSIM_FN = "./results-%s/test_SSIMs-%s.txt" % (VERSION, VERSION)
LOGS_DIR_SELECT = "%s/select/" % LOGS_DIR
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
if not os.path.exists(LOGS_DIR_SELECT):
    os.makedirs(LOGS_DIR_SELECT)
if not os.path.exists(IMGS_DIR):
    os.makedirs(IMGS_DIR)

# define parameters for test
EXPORT_RESULT_IMAGES = True
EXPORT_8BIT_PNG = False
EXPORT_16BIT_TIFF = True

# define hyperparameters for training
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 #
NUM_EPOCHS = 500 #
RENORM_SCALE = 1.0 
RENORMALIZE_INPUT = True # 

# interleaving rate
INTERLEAVE_RATE = 2

# training loss function
USE_LOG_L2_BOTH = True 
USE_LOG_L1_BOTH = False

# view grid
numViewsYX = [9,9]
numViews = numViewsYX[0]*numViewsYX[1]
HalfViewY, HalfViewX = (numViewsYX[0]-1)//2, (numViewsYX[1]-1)//2

# input views
if (numInputViews==9):
   inputView_list = [0, HalfViewX, numViewsYX[1]-1, 
                     HalfViewY*numViewsYX[1], (numViews-1)//2, HalfViewY*numViewsYX[1]+numViewsYX[1]-1, 
                     (numViewsYX[0]-1)*numViewsYX[1], (numViewsYX[0]-1)*numViewsYX[1]+HalfViewX, numViews-1] 
elif (numInputViews==5):
   inputView_list = [0, numViewsYX[1]-1, (numViews-1)//2, (numViewsYX[0]-1)*numViewsYX[1], numViews-1]
else:
   input("you need to define the input view list")

# decide what data to load
LOAD_RGBD = True

# option to compute quality metrics
REPORT_QUALITY = False # True

# other parameters
imH, imW = 512, 512 # size of image
if MODE=="TEST":
   H, W = imH, imW # size of each image at test
else:
   H, W = 128, 128 # size of each input patch at training

C = 3 # number of color channels
depthScale = 12 # divide input depth map by this value # should maintain global relative depth 
diopterScale = 2.2 # divide input diopter map by this value # should maintain global relative diopter 
crop_width = 0 # crop on each side
lookAt = 0 # meter

if MODE=="TRAIN":
   numEvalScenes = endEvalSceneIDX-startEvalSceneIDX+1
   print("number of validation scenes = %d" % numEvalScenes)
   numTrainScenes = endSceneIDX-startSceneIDX+1
   print("number of training scenes = %d" % numTrainScenes)
   numSamplesPerScene = 49*C # treat each color channel as separate sample
   numTrainSamples = numTrainScenes*numSamplesPerScene
   print("number of training samples = %d" % numTrainSamples)
   numTrainSamplesPerIter = numTrainScenesPerIter*numSamplesPerScene
   NUM_ITER_PER_EPOCH = numTrainScenes // numTrainScenesPerIter
   print("number of iterations per echo = %d" % NUM_ITER_PER_EPOCH)
   PERIOD = numTrainSamplesPerIter//BATCH_SIZE 
   print("number of period per iteration = %d" % PERIOD)
   EVAL_STEP = PERIOD
   VIS_STEP = 10


#---------------------------------------------------------------------------------------------
# define network
INPUT_DIM = 2*numInputViews # intensity + depth for each view
OUTPUT_DIM = numViewsYX[0]*numViewsYX[1] - numInputViews # intensity for novel views

# tf Graph input (only pictures)
X = tf.placeholder("float", [None,INPUT_DIM*(INTERLEAVE_RATE**2),H//INTERLEAVE_RATE,W//INTERLEAVE_RATE]) 
Y = tf.placeholder("float", [None,OUTPUT_DIM,H,W])

weightVarScale = 0.25 
activationFunc = tf.nn.elu 
bias_stddev = 0.01

def model(x_in):
    # layer parameters::
    L = 12 # number of layers 
    fwConstant = 3
    fnumConstant = 128
    fw = np.full((L), fwConstant, dtype=int)
    fnum = np.append(np.full((L-1), fnumConstant, dtype=int), OUTPUT_DIM*(INTERLEAVE_RATE**2))

    layers_params = {}
    layers = {}
    prev_layers = {}


    for i in range(0, L):    
        if i==0: # first layer
           in_dim, out_dim = INPUT_DIM*(INTERLEAVE_RATE**2), fnum[i]
        elif i==L-1: # last layer
           in_dim, out_dim = fnum[i-1]+numInputViews*(INTERLEAVE_RATE**2), OUTPUT_DIM*(INTERLEAVE_RATE**2) #include skip connection for light field application
        else:
           in_dim, out_dim = fnum[i-1], fnum[i]  

        layers_params[i] = {'weights':init_weights([fw[i],fw[i],in_dim, out_dim], 'xavier',xavier_params=(in_dim, out_dim),r=weightVarScale),
                     'bias':tf.Variable(tf.truncated_normal([out_dim],stddev=bias_stddev))}

    # build layers::
    print("input data:", x_in.shape)
    if RENORMALIZE_INPUT:
       x_in = (x_in - 0.5*RENORM_SCALE) 

    for i in range(0, L):
        if i==0:
            prev_layers[i] = x_in   
        elif (i<3) or (i%2==0): 
            prev_layers[i] = layers[i-1]
        else: 
            prev_layers[i] = layers[i-1] + layers[i-3]
            print('(skip connection: %d, %d)'%(i-1, i-3))

        ## include skip connection
        if i==L-1:
            if RENORMALIZE_INPUT:
                skip_in = 2*x_in[:,::2,:,:] # make it to range (-1,1)
            else:
                skip_in = 2*x_in[:,::2,:,:] - 1.0
            prev_layers[i] = tf.concat([prev_layers[i], skip_in], axis=1)
            print('(skip connection before %d)' % i)

        if i==L-1:
           layers[i] = tf.tanh(tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.conv2d(prev_layers[i],layers_params[i]['weights'],strides=[1,1,1,1], padding='SAME', data_format='NCHW'), layers_params[i]['bias'], data_format='NCHW'),axis=1))    
        else:
           layers[i] = activationFunc(tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.conv2d(prev_layers[i],layers_params[i]['weights'],strides=[1,1,1,1], padding='SAME', data_format='NCHW'), layers_params[i]['bias'], data_format='NCHW'),axis=1))    
        print("layer %d:" % i, layers[i].shape)
             
    # renormalize to desired scale    
    x_out = tf.add(0.5, tf.scalar_mul(0.5, layers[L-1]), name='x_out')     
    print("output tensor:", x_out.shape)

    return deinterleave(INTERLEAVE_RATE, x_out)

# construct model
model_op =  model(X)


# define loss and optimizer
labels, predictions = Y[:,0:OUTPUT_DIM,:,:], model_op

ssim_intensity = tf.reduce_mean(tf.image.ssim(tf.transpose(labels, [0,2,3,1]), tf.transpose(predictions, [0,2,3,1]), max_val = 1.0))
rmse_intensity = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
psnr_intensity = 20*log10(RENORM_SCALE) - 10*log10(rmse_intensity)


if USE_LOG_L2_BOTH: 
   labels_dx, labels_dy = calImageGradients(labels)
   preds_dx, preds_dy = calImageGradients(predictions)
   rmse_grad_x, rmse_grad_y = tf.losses.mean_squared_error(labels=labels_dx, predictions=preds_dx), tf.losses.mean_squared_error(labels=labels_dy, predictions=preds_dy)
   psnr_grad_x, psnr_grad_y = -20*log10(RENORM_SCALE) + 10*log10(rmse_grad_x), -20*log10(RENORM_SCALE) + 10*log10(rmse_grad_y)
   loss = -psnr_intensity + 0.5*(psnr_grad_x + psnr_grad_y)
elif USE_LOG_L1_BOTH:
   log_diff_intensity = log10(tf.reduce_mean(tf.abs(labels-predictions))) 
   labels_dx, labels_dy = calImageGradients(labels)
   preds_dx, preds_dy = calImageGradients(predictions)
   log_diff_grad_x = log10(tf.reduce_mean(tf.abs(labels_dx-preds_dx)))
   log_diff_grad_y = log10(tf.reduce_mean(tf.abs(labels_dy-preds_dy)))
   loss = log_diff_intensity + 0.5*(log_diff_grad_x + log_diff_grad_y)   


# training optimizer
optimizer =  tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss=loss) 
grads_and_vars = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999, epsilon=1e-8).compute_gradients(loss, tf.trainable_variables()) 

# set up saver
saver = tf.train.Saver(max_to_keep=5)

# initialization
init = tf.global_variables_initializer()

# create a summary to monitor cost tensor
training_summary  = tf.summary.scalar("training loss", loss, family=TF_SUMMARY_FLAG)
validation_summary  = tf.summary.scalar("validation loss", loss, family=TF_SUMMARY_FLAG)

# start training
with tf.Session() as sess:    
   
    # run the initialization
    sess.run(init)  
    
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIR, graph=tf.get_default_graph())
    
    # restore model if trained 
    if RESTORE_TRAINED_MODEL or MODE!="TRAIN":
        ckpt = tf.train.get_checkpoint_state(RESTORE_LOGS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, tf.train.latest_checkpoint(RESTORE_LOGS_DIR))
            print("model restored from %s" % RESTORE_LOGS_DIR)
        else:
            input("no restored model...")

    min_ell_eval_mean = []

    #-------------------------------------------------------------------------------
    # training
    if MODE == "TRAIN":
        print("training mode")

        # load validation data used at training
        idxsScenes_eval = np.arange(startEvalSceneIDX, endEvalSceneIDX+1)
        numEvalSamples, cleanRGB_eval, diopter_eval = loadData(DATA_PATH, H,W,imH,imW,numViewsYX,idxsScenes_eval,C,depthScale, diopterScale, LOAD_RGBD)

        # load training data used at this iteration
        idxsScenes = np.random.permutation(numTrainScenes) + startSceneIDX
        idxsScenes = idxsScenes[0:numTrainScenesPerIter]
        numTrainSamples, cleanRGB, diopter = loadData(DATA_PATH, H,W,imH,imW,numViewsYX,idxsScenes,C,depthScale, diopterScale, LOAD_RGBD)


        for e in range(0, NUM_EPOCHS):
          print("--train epoch:", e)

          for iter in range(NUM_ITER_PER_EPOCH):
                print("----train iter:", iter)

                # generate batch indices
                idxs = np.random.permutation(numTrainSamples) # shuffle all samples

                ii = 0
                for i in range(PERIOD):
                    # generate current batch
                    batch_x = np.zeros((BATCH_SIZE,INPUT_DIM,H,W))
                    batch_y = np.zeros((BATCH_SIZE,OUTPUT_DIM,H,W))
                    for b in range(BATCH_SIZE):
                        for iv in range(numInputViews):
                            batch_x[b,iv*2,:,:] = cleanRGB[idxs[ii],:,:,inputView_list[iv]].reshape((1,1,H,W))
                            batch_x[b,iv*2+1,:,:] = diopter[idxs[ii],:,:,inputView_list[iv]].reshape((1,1,H,W))
                        kk = 0
                        for vv in range(0, numViews):
                            if not (vv in inputView_list):
                               batch_y[b,kk,:,:] = cleanRGB[idxs[ii],:,:,vv].reshape((1,1,H,W))
                               kk = kk + 1
                        
                        ii = ii + 1              
                            
                    batch_x_reshape = interleave_np(INTERLEAVE_RATE, batch_x) #use numpy version
                    
                    _, ell, train_summ = sess.run([optimizer, loss, training_summary],feed_dict={X:batch_x_reshape, Y:batch_y})
 
                    # Write logs at every iteration
                    summary_writer.add_summary(train_summ, e*NUM_ITER_PER_EPOCH*PERIOD+iter*PERIOD+i) 
          
                    # display logs per step
                    if (i==1) or (i==PERIOD-1) or (i % VIS_STEP==0):
                       print("training epoch %d, iteration %d, period %d: batch loss (train): %f " % (e, iter, i, ell))

                    # validation # e in range(NUM_EPOCHS):
                    if (i==PERIOD-1) or (i>0 and (i % EVAL_STEP==0)): 
                        # save current model trained 
                        if numEvalSamples==0:
                           saver.save(sess, LOGS_DIR + "model.ckpt", e) 

                        if numEvalSamples>0:
                            ell_eval_mean = []
                            ell_psnr_eval_mean = []
                            ell_ssim_eval_mean = []

                            for j in range(0,numEvalSamples-BATCH_SIZE,BATCH_SIZE):
                                idxs_eval = np.linspace(start=j, stop=j+BATCH_SIZE-1, num=BATCH_SIZE, dtype=np.int16)  # for valiation use, no randomness
                                batch_x_eval = np.zeros((BATCH_SIZE,INPUT_DIM,H,W))
                                batch_y_eval = np.zeros((BATCH_SIZE,OUTPUT_DIM,H,W))
                                for be in range(BATCH_SIZE): 
                                    for iv in range(numInputViews):
                                        batch_x_eval[be,iv*2,:,:] = cleanRGB_eval[idxs_eval[be],:,:,inputView_list[iv]].reshape((1,1,H,W))
                                        batch_x_eval[be,iv*2+1,:,:] = diopter_eval[idxs_eval[be],:,:,inputView_list[iv]].reshape((1,1,H,W))
                                    kk = 0
                                    for vv in range(0, numViews):
                                        if not (vv in inputView_list):
                                           batch_y_eval[be,kk,:,:] = cleanRGB_eval[idxs_eval[be],:,:,vv].reshape((1,1,H,W)) 
                                           kk = kk + 1                          
                        
                                #
                                batch_x_eval_reshape = interleave_np(INTERLEAVE_RATE, batch_x_eval) #use numpy version

                                # run evauation
                                ell_eval, ell_psnr_eval, ell_ssim_eval = sess.run([loss, psnr_intensity, ssim_intensity], feed_dict={X:batch_x_eval_reshape, Y:batch_y_eval})
                                ell_eval_mean.append(ell_eval)
                                ell_psnr_eval_mean.append(ell_psnr_eval)
                                ell_ssim_eval_mean.append(ell_ssim_eval)
                            
                            validation_summ = tf.Summary()
                            ell_eval_mean = np.mean(ell_eval_mean)                        
                            validation_summ.value.add(tag="validation loss (%s)" % TF_SUMMARY_FLAG, simple_value=ell_eval_mean)
                            print("training epoch %d, period %d, loss (validate): %f" % (e, i, ell_eval_mean))

                            ell_psnr_eval_mean = np.mean(ell_psnr_eval_mean)                        
                            validation_summ.value.add(tag="validation psnr intensity (%s)" % TF_SUMMARY_FLAG, simple_value=ell_psnr_eval_mean)
                            print("training epoch %d, period %d, psnr intensity (validate): %f" % (e, i, ell_psnr_eval_mean))

                            ell_ssim_eval_mean = np.mean(ell_ssim_eval_mean)                        
                            validation_summ.value.add(tag="validation ssim (%s)" % TF_SUMMARY_FLAG, simple_value=ell_ssim_eval_mean)
                            print("training epoch %d, period %d, ssim intensity (validate): %f" % (e, i, ell_ssim_eval_mean))

                            # write logs for validation result
                            summary_writer.add_summary(validation_summ, e*NUM_ITER_PER_EPOCH*PERIOD+iter*PERIOD+i)
                                        
                            if (e==0) or (ell_eval_mean<min_ell_eval_mean):
                                min_ell_eval_mean = ell_eval_mean
                                saver.save(sess, LOGS_DIR_SELECT + "model-best.ckpt", e*NUM_ITER_PER_EPOCH*PERIOD+iter*PERIOD+i) 
                            #----------------------------------------------------------------------

                    
        print("training is done!")
        print("Run the command line in cmd:\n" \
              "--> tensorboard --logdir=%s " % LOGS_DIR)
        print("The best model is stored at: %s" % LOGS_DIR_SELECT)
    
    # testing
    if MODE == "TEST":
        print("test mode")
        idxsScenes_test = np.arange(startSceneIDX, endSceneIDX+1)
        numTotalSamples, cleanRGB, diopter = loadData(DATA_PATH, H,W,imH,imW,numViewsYX,idxsScenes_test,C,depthScale, diopterScale, LOAD_RGBD)
        
        test_PSNRs = []
        test_SSIMs = []
        recon_fs_rgb = np.zeros((C,H,W,OUTPUT_DIM), dtype=np.float32)
        batch_x = np.zeros((C,INPUT_DIM,H,W), dtype=np.float32)
        batch_y = np.zeros((C,OUTPUT_DIM,H,W), dtype=np.float32)

        for i in range(0, numTotalSamples//C, 1):
            for c in range(0, C):
                # generate current batch
                for iv in range(numInputViews):
                    batch_x[c,iv*2,:,:] = cleanRGB[i*C+c,:,:,inputView_list[iv]].reshape((1,1,H,W))
                    batch_x[c,iv*2+1,:,:] = diopter[i*C+c,:,:,inputView_list[iv]].reshape((1,1,H,W))
                kk = 0
                for vv in range(0, numViews):
                    if not (vv in inputView_list):
                       batch_y[c,kk,:,:] = cleanRGB[i*C+c,:,:,vv].reshape((1,1,H,W))  
                       kk = kk + 1
                
                batch_x_reshape = interleave_np(INTERLEAVE_RATE, batch_x) #use numpy version
                
                # run evauation
                time_start = time.clock()
                recon_fs = sess.run(model_op, feed_dict={X:batch_x_reshape[c,:,:,:].reshape((1,INPUT_DIM*(INTERLEAVE_RATE**2),H//INTERLEAVE_RATE,W//INTERLEAVE_RATE)), Y:batch_y[c,:,:,:].reshape((1,OUTPUT_DIM,H,W))})
                print(time.clock() - time_start)

                # save
                recon_fs_rgb[c,:,:,:] = np.transpose(recon_fs.reshape((1,OUTPUT_DIM,H,W)), (0,2,3,1))



            # save image files
            if EXPORT_RESULT_IMAGES:
                vv, ww = 0, 0    
                for viewY in range(numViewsYX[0]):
                   for viewX in range(numViewsYX[1]):
                      view = viewY*numViewsYX[1]+viewX
                      im_input = np.zeros((H,W,C))
                      im_recon = np.zeros((H,W,C))        

                      if (view in inputView_list):
                         for c in range(0,C):
                             im_input[:,:,c] = np.clip(batch_x[c,ww,:,:].reshape((H,W)),0,1)
                         ww = ww + 2
                      else: # not input views of network
                         for c in range(0,C):
                           im_input[:,:,c] = np.clip(batch_y[c,vv,:,:].reshape((H,W)),0,1)    
                           im_recon[:,:,c] = np.clip(recon_fs_rgb[c,:,:,vv].reshape((H,W)),0,1)  
                         vv = vv + 1
                         
                         if EXPORT_8BIT_PNG:
                            fn = "%s/recon_lf%03d_im%02d_%02d.png" % (IMGS_DIR,idxsScenes_test[i],viewY,viewX)
                            save_png(im_recon, fn)      
                         elif EXPORT_16BIT_TIFF:
                            fn = "%s/recon_lf%03d_im%02d_%02d.tiff" % (IMGS_DIR,idxsScenes_test[i],viewY,viewX)           
                            save_tiff(im_recon, fn)

                      #-----------------------------------
                      if EXPORT_8BIT_PNG:
                         fn = "%s/input_lf%03d_im%02d_%02d.png" % (IMGS_DIR,idxsScenes_test[i],viewY,viewX)
                         save_png(im_input, fn)     
                      elif EXPORT_16BIT_TIFF:
                         fn = "%s/input_lf%03d_im%02d_%02d.tiff" % (IMGS_DIR,idxsScenes_test[i],viewY,viewX)           
                         save_tiff(im_input, fn)
                                              
                      
            # compute and save PSNR
            if REPORT_QUALITY:   
               print("calculating PSNR and SSIM...")
               psnrs = cal_psnr_focalstack(recon_fs_rgb/RENORM_SCALE, np.transpose(batch_y[:,0:OUTPUT_DIM,:,:],(0,2,3,1))/RENORM_SCALE,crop_width)
               test_PSNRs.append(psnrs)
               ssims = cal_ssim_focalstack(recon_fs_rgb/RENORM_SCALE, np.transpose(batch_y[:,0:OUTPUT_DIM,:,:],(0,2,3,1))/RENORM_SCALE,crop_width)
               test_SSIMs.append(ssims.eval()) #need to eval because it uses tensorflow functions
 
        if REPORT_QUALITY:                
            test_PSNRs = np.array(test_PSNRs)
            mean_PSNR_all = np.mean(test_PSNRs)
            mean_PSNR_each = np.mean(test_PSNRs, axis=1)
            myFile = open(PSNR_FN, 'wb')
            np.savetxt(myFile, mean_PSNR_all.reshape((1,1)), fmt='%.4f',header="\nmean PSNR for all test images")
            np.savetxt(myFile, mean_PSNR_each, fmt='%.4f',header="\nmean PSNR for each focal stack")
            np.savetxt(myFile, test_PSNRs, fmt='%.4f',header="\nPSNR for each individual image")
        
            myFile.close()        
            print("mean_PSNR_all:", mean_PSNR_all)
            print("mean_PSNR_each:\n", mean_PSNR_each)        
            print("testing is done!")

            test_SSIMs = np.array(test_SSIMs)
            mean_SSIM_all = np.mean(test_SSIMs)
            mean_SSIM_each = np.mean(test_SSIMs, axis=1)
            myFile = open(SSIM_FN, 'wb')
            np.savetxt(myFile, mean_SSIM_all.reshape((1,1)), fmt='%.4f',header="\nmean SSIM for all test images")
            np.savetxt(myFile, mean_SSIM_each, fmt='%.4f',header="\nmean SSIM for each focal stack")
            np.savetxt(myFile, test_SSIMs, fmt='%.4f',header="\nSSIM for each individual image")

            myFile.close()        
            print("mean_SSIM_all:", mean_SSIM_all)
            print("mean_SSIM_each:\n", mean_SSIM_each)     
            
        print("test is done!")

