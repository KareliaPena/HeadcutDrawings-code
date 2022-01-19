"""
Created on Mon Nov  5 19:42:49 2018

@author: kareliap
"""

from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import scipy.io as sio
import scipy
import math
import os

from datetime import datetime


tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


#%%
def XDoG_Dots(inputIm):
    inputIm = cv2.cvtColor(inputIm, cv2.COLOR_RGB2GRAY)
#    inputIm = cv.imread(input_img,0)
    Tao = 0.94
    Phi = 40
    Epsilon = 0.0001
    k = 1.4
    Sigma = 1.1
    size=int(2*np.ceil(2*Sigma)+1)
    size1=int(2*np.ceil(2*k*Sigma)+1)
    gFilteredIm1 = cv2.GaussianBlur(inputIm,(size,size),Sigma)
    gFilteredIm2 = cv2.GaussianBlur(inputIm,(size1,size1),Sigma * k,Sigma * k)

    differencedIm2 = gFilteredIm1 - (Tao * gFilteredIm2)

    x = differencedIm2.shape[0]
    y = differencedIm2.shape[1]
    
    #Extended difference of gaussians
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] >= Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 1 + np.tanh(Phi*((differencedIm2[i, j]-Epsilon)));
    
    differencedIm2 = np.double(differencedIm2)
    out = np.zeros(differencedIm2.shape, np.double)
    normalized = cv2.normalize(differencedIm2, out, 1.0, 0.0, cv2.NORM_MINMAX)            
    
    normalized=(255*normalized).astype('uint8')
#    cv2.imshow('normalized',normalized)
    
    return normalized


#%%
def XDoG_Outline(inputIm):
#    inputIm = cv.imread(input_img,0)
    inputIm = cv2.cvtColor(inputIm, cv2.COLOR_RGB2GRAY)

    Tao = 0.998
    Phi = 200
    Epsilon = 0.001
    k = 1.6
    Sigma = 1.3
    size=int(2*np.ceil(2*Sigma)+1)
    size1=int(2*np.ceil(2*k*Sigma)+1)
    gFilteredIm1 = cv2.GaussianBlur(inputIm,(size,size),Sigma)
    gFilteredIm2 = cv2.GaussianBlur(inputIm,(size1,size1),Sigma * k,Sigma * k)

    differencedIm2 = gFilteredIm1 - (Tao * gFilteredIm2)

    x = differencedIm2.shape[0]
    y = differencedIm2.shape[1]
    
    #Extended difference of gaussians
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] >= Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 1 + np.tanh(Phi*((differencedIm2[i, j]-Epsilon)));
    
    differencedIm2 = np.double(differencedIm2)
    out = np.zeros(differencedIm2.shape, np.double)
    normalized = cv2.normalize(differencedIm2, out, 1.0, 0.0, cv2.NORM_MINMAX)            
    mean=np.mean(normalized)
    ret,img_in = cv2.threshold(normalized,mean,1.0,cv2.THRESH_BINARY)
    img_in=(255*(img_in)).astype('uint8')
#    cv2.imshow('img_in',img_in)
    return img_in


#%%


def load_mask3(mask_path,content_path,th):
  img = np.squeeze(load_img_Gray(mask_path))  
  ImC =  np.squeeze(load_img_Gray(content_path))  + 170
  
  ImC = ImC/ImC.max()*220
   # Find width and height of image
  row, column = img.shape
  # Create an zeros array to store the sliced image
  img1 = np.zeros((row,column),dtype = 'uint8')
  # Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
  for i in range(row):
      for j in range(column):
          if img[i,j]<(ImC[i,j]):
              img1[i,j] = 255
          if img[i,j]>(ImC[i,j]):
              img1[i,j] = 0
  thresh1=img1
  #ret,thresh1 = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
  thresh1= np.expand_dims(np.expand_dims(thresh1, axis=0), axis=3)
  final = thresh1
  final = np.append(final, thresh1,axis=3)
  final = np.append(final, thresh1,axis=3)
  imshow(final, title='Binary Mask')
  
  img = tf.keras.applications.vgg19.preprocess_input(final.astype('float32'))
  
  
  return img , final

def load_img_Gray(path_to_img):
  img = np.expand_dims(cv2.imread(path_to_img,0),axis=2)
  long = max(img.shape)
  img = kp_image.array_to_img(img)
  
  img = scipy.misc.imresize(img, (content_height,content_width))
  
  img = kp_image.img_to_array(img)
  return img

def init_mix_image(content_path,mask_path,alpha):
  model = 'model.yml.gz'
  img1=load_img(content_path)
  _,img2 = load_mask3(mask_path,content_path,255/2-50)
  im=np.squeeze(img1)
  edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
  rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)  
  edges = 1/edges.max()*edges
  edges = np.expand_dims(np.expand_dims(edges, axis=0), axis=3)
  img=((1-edges)*img2+edges*img1)*(1-alpha)+alpha*img1
  total = tf.keras.applications.vgg19.preprocess_input(img)
  
  return total,edges,img
    
def plotGramMatrix(units,title=None):
    dim = units.shape
    filters = dim[3]
    plt.figure(num=None, figsize=(20,20))
    n_rows = 2
    n_columns = math.ceil(filters / n_rows) + 1
    if title is not None:
     plt.suptitle(title)
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="cool")
        plt.colorbar()

    
def plotNNFilter(units,title=None):
    dim = units.get_shape().as_list()
    filters = dim[3]
    plt.figure(num=None, figsize=(20,20))
    n_rows = 8
    n_columns = math.ceil(filters / n_rows) + 1
    if title is not None:
     plt.suptitle(title)
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    


def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
    color_codes = ['BLACK','BLUE', 'GREEN', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']
    def _extract_mask(seg, color_str):
        h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
            
        #aux = np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)
        #kernel = np.ones((5,5),np.uint8)
        #dilation = cv2.dilate(aux,kernel,iterations = 1)
        
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32) # this is an AND among the three channels

    # PIL resize has different order of np.shape
    content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0
    style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0

    color_content_masks = []
    color_style_masks = []
    for i in range(len(color_codes)):
        color_content_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(content_seg, color_codes[i])), 0), -1))
        color_style_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(style_seg, color_codes[i])), 0), -1))

    return color_content_masks, color_style_masks

def gram_matrix(activations):
    height = tf.shape(activations)[1]
    width = tf.shape(activations)[2]
    num_channels = tf.shape(activations)[3]
    gram_matrix = tf.transpose(activations, [0, 3, 1, 2]) #perm: A permutation of the dimensions of a.
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix





  #style_score_list, style_mask_score_list= segment_style_loss(model, layers_names, style_layers, mask_layers, style_features, style_output_features_var,mask_features, mask_output_features_var, content_masks, style_masks[1:], float(1e2))


def segment_style_loss(CNN_structure, layers_names, style_layers,maks_style_layers, const_layers, var_layers, mask_features, mask_output_features_var,content_segs, style_segs, weight,weight_mask=1):
    loss_styles = []
    loss_mask_styles = []
    layer_count = float(len(const_layers))
    layer_index = 0
    layer_index_mask = 0
    
    _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
    _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
    for i in range(len(layers_names)):
        layer_name = layers_names[i]

        # downsampling segmentation
        if "pool" in layer_name:
            content_seg_width, content_seg_height = int(math.floor(content_seg_width / 2)), int(math.floor(content_seg_height / 2))
            style_seg_width, style_seg_height = int(math.floor(style_seg_width / 2)), int(math.floor(style_seg_height / 2))

            for i in range(len(content_segs)):
                content_segs[i] = tf.image.resize_bilinear(content_segs[i], tf.constant((content_seg_height, content_seg_width)))
                style_segs[i] = tf.image.resize_bilinear(style_segs[i], tf.constant((style_seg_height, style_seg_width)))

        elif "conv" in layer_name:
            for i in range(len(content_segs)):
                # have some differences on border with torch
                content_segs[i] = tf.nn.avg_pool(tf.pad(content_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                style_segs[i] = tf.nn.avg_pool(tf.pad(style_segs[i], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                kernel = np.ones((5,5),np.uint8)
                #content_segs[i]= tf.constant(np.expand_dims(np.expand_dims(cv2.dilate(np.squeeze(content_segs[i]) ,kernel,iterations = 1),2),0))
                #style_segs[i]=tf.constant(np.expand_dims(np.expand_dims(cv2.dilate(np.squeeze(style_segs[i]) ,kernel,iterations = 1),2),0))

        if layer_name in style_layers:
            #print("Setting up style layer: <{}>".format(layer_name))
            const_layer = const_layers[layer_index]
            var_layer = var_layers[layer_index]

            layer_index = layer_index + 1

            layer_style_loss = 0.0
            gram_matrices_var = None
            gram_matrices_const = None
            cont = 0
            for content_seg, style_seg in zip(content_segs[1:], style_segs[1:]):
               
                gram_matrix_const = gram_matrix(tf.multiply(const_layer, style_seg))
                Const_FeatureMaps = tf.multiply(const_layer, style_seg)
                style_mask_mean   = tf.reduce_mean(style_seg)
                gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                        lambda: gram_matrix_const / (tf.to_float(tf.size(const_layer)) * style_mask_mean),
                                        lambda: gram_matrix_const
                                    )
                #https://www.tensorflow.org/api_docs/python/tf/cond
                gram_matrix_var   = gram_matrix(tf.multiply(var_layer, content_seg)) # Multiplication of each feature map by the segmentation mask 
                Var_FeatureMaps = tf.multiply(var_layer, content_seg)
                content_mask_mean = tf.reduce_mean(content_seg)   # The mean indicates the amount of values different from zero in the segmented mask 
                gram_matrix_var   = tf.cond(tf.greater(content_mask_mean, 0.),
                                        lambda: gram_matrix_var / (tf.to_float(tf.size(var_layer)) * content_mask_mean), # it is a ponderated sum so it is divided by the number of elements different from zero in the segmented mask 
                                        lambda: gram_matrix_var # in this case should be zero
                                    )

                diff_style_sum    = tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean  # It should be multiplied by the nonzero positions on 
                
                layer_style_loss += diff_style_sum
                cont = cont+1
                #plotNNFilter(Const_FeatureMaps,'Const_FeatureMaps' + str(cont))
                #plotNNFilter(Var_FeatureMaps,'Var_FeatureMaps' + str(cont))
            loss_styles.append(layer_style_loss * weight)
        
        if layer_name in maks_style_layers:
            #print("Setting up style layer: <{}>".format(layer_name))
            const_layer = mask_features[layer_index_mask]
            var_layer = mask_output_features_var[layer_index_mask]

            layer_index_mask = layer_index_mask + 1

            layer_mask_style_loss = 0.0
            cont = 0
            style_seg = 1-content_segs[0]
            content_seg = 1-content_segs[0]
            #style_seg =content_segs[2]
            #content_seg =content_segs[2]
            
            gram_matrix_const = gram_matrix(tf.multiply(const_layer, style_seg))
            Const_FeatureMaps = tf.multiply(const_layer, style_seg)
            style_mask_mean   = tf.reduce_mean(style_seg)
            gram_matrix_const = tf.cond(tf.greater(style_mask_mean, 0.),
                                    lambda: gram_matrix_const / (tf.to_float(tf.size(const_layer)) * style_mask_mean),
                                    lambda: gram_matrix_const
                                )
            #https://www.tensorflow.org/api_docs/python/tf/cond
            gram_matrix_var   = gram_matrix(tf.multiply(var_layer, content_seg)) # Multiplication of each feature map by the segmentation mask 
            Var_FeatureMaps = tf.multiply(var_layer, content_seg)
            content_mask_mean = tf.reduce_mean(content_seg)   # The mean indicates the amount of values different from zero in the segmented mask 
            gram_matrix_var   = tf.cond(tf.greater(content_mask_mean, 0.),
                                    lambda: gram_matrix_var / (tf.to_float(tf.size(var_layer)) * content_mask_mean), # it is a ponderated sum so it is divided by the number of elements different from zero in the segmented mask 
                                    lambda: gram_matrix_var # in this case should be zero
                                )

            diff_style_sum    = tf.reduce_mean(tf.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean  # It should be multiplied by the nonzero positions on 
            
            layer_mask_style_loss += diff_style_sum
      
                #plotNNFilter(Const_FeatureMaps,'Const_FeatureMaps' + str(cont))
                #plotNNFilter(Var_FeatureMaps,'Var_FeatureMaps' + str(cont))
            loss_mask_styles.append(layer_mask_style_loss * weight_mask)    
            
            
    return loss_styles, loss_mask_styles  


def load_mask(mask_path,th):
  img = cv2.imread(mask_path,0)
  ret,thresh1 = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
  auxt=np.expand_dims(thresh1, axis=0)
  #imshow(auxt, title='Binary Mask')
  thresh1=np.expand_dims(np.expand_dims(thresh1, axis=0), axis=3)
  
  thresh1 = tf.image.resize_bilinear(thresh1, tf.constant((content_height, content_width)))
  
  return thresh1


def load_mask2(mask_path,th):
  img = cv2.imread(mask_path,0)
  ret,thresh1 = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
  #imshow(auxt, title='Binary Mask')
  thresh1= np.expand_dims(np.expand_dims(thresh1, axis=0), axis=3)
  final = thresh1
  final = np.append(final, thresh1,axis=3)
  final = np.append(final, thresh1,axis=3)
  img = tf.keras.applications.vgg19.preprocess_input(final.astype('float32'))
  
  img = tf.image.resize_bilinear(img, tf.constant((content_height, content_width)))
  final=tf.image.resize_bilinear(final, tf.constant((content_height, content_width)))
  
  return img , final


def load_img(path_to_img):
  max_dim = 900
  #img = Image.open(path_to_img)
  img = np.expand_dims(cv2.imread(path_to_img,0),axis=2)
  
  long = max(img.shape)
  #max_dim = max(img.shape) 
  #img = kp_image.img_to_array(img)
  #len(img.size)<3
  if len(img.shape)==3:
      imgt = img
      imgt = np.append(imgt, img,axis=2)
      imgt = np.append(imgt, img,axis=2)
      img = imgt
  img = kp_image.array_to_img(img)
  scale = max_dim/long
  
  img = scipy.misc.imresize(img, scale)
  #img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img


def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out,'gray')
  

def load_and_process_img_difSize(path_to_img,sizeSt):
  max_dim = int(sizeSt)
  #img = Image.open(path_to_img)
  img = np.expand_dims(cv2.imread(path_to_img,0),axis=2)
  
  long = max(img.shape)
  #img = kp_image.img_to_array(img)
  #len(img.size)<3
  if len(img.shape)==3:
      imgt = img
      imgt = np.append(imgt, img,axis=2)
      imgt = np.append(imgt, img,axis=2)
      img = imgt
  img = kp_image.array_to_img(img)
  scale = max_dim/long
  img = scipy.misc.imresize(img, scale)
  #img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img 
    

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  #img = 255/img.max()*img
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img


def load_init_noisy_image(h,w):
  init_image = 255*(np.random.rand(h,w).astype('float32'))
  init_image = np.expand_dims(init_image, axis=2)
  imgt = init_image
  imgt = np.append(imgt, init_image,axis=2)
  imgt = np.append(imgt, init_image,axis=2)
  init_image = imgt
  init_image = np.expand_dims(init_image, axis=0)
  init_image = tf.keras.applications.vgg19.preprocess_input(init_image)
  return init_image

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, axis=0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x


def deprocess_img2(processed_img):
  #x = processed_img
   # perform the inverse of the preprocessiing step
   
  x = np.squeeze(processed_img[:,:, :, 0] + 103.939,0)
  x = np.expand_dims(x,axis=2)
  x = np.append(x,np.expand_dims(np.squeeze(processed_img[:,:, :, 1] + 116.779,0),2),axis=2)
  x = np.append(x,np.expand_dims(np.squeeze(processed_img[:,:, :, 2] + 123.68,0),2),axis=2)
  
  x =np.expand_dims(x,axis=0)

  #x = x[:, :, ::-1]

  return x


def get_model():
 
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  layers = vgg.layers
  # Get output layers corresponding to style and content layers 
  model_outputs = [vgg.get_layer(layers[i].name).output for i in range(len(layers))]
  layers_names = [layers[i].name for i in range(len(layers))]
  # Build model 
  return models.Model(vgg.input, model_outputs[1:]), layers_names[1:]


def get_used_layers_model():

  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target,edges):
  edges2 = tf.image.resize_bilinear(edges, tf.constant((target.shape[1], target.shape[2])))
  return tf.reduce_mean(tf.square(tf.multiply(tf.expand_dims(base_content,0) - target,edges2)))

def get_Binary_Mask_loss(base_content, Grid_image,Black_Mask):
  return tf.reduce_mean(tf.square(tf.multiply(tf.multiply(Grid_image,1-Black_Mask),255-base_content)))



def get_style_loss(base_style, target_style,content_masks):

  gram_style = gram_matrix(base_style)
  gram_target = gram_matrix(target_style)
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)




def get_feature_representations(model, layers_names, content_layers, style_layers, mask_layers, content_path, style_path,content_seg_path,style_seg_path,mask_path):
 
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img_difSize(style_path,sizeSt)
  
  content_width, content_height = content_image.shape[2], content_image.shape[1]
  
  mask_image, _ = load_mask3(mask_path,content_path,255/2-50)
  mask_image = np.array(mask_image)
  
  style_width, style_height = style_image.shape[2], style_image.shape[1]
  content_width, content_height = content_image.shape[2], content_image.shape[1]
  
  content_masks, style_masks = load_seg(content_seg_path, style_seg_path, [content_width, content_height], [style_width, style_height])

  mask_image_no_background=mask_image.copy()
  auxx=content_masks[0]
  auxx = np.append(auxx, content_masks[0],axis=3)
  auxx = np.append(auxx, content_masks[0],axis=3)
  
  mask_image_no_background[auxx>0.5]=content_image[auxx>0.5]
  
  #aux[auxx==1]=mask_image.max()
  
  # batch compute content and style features
  mask_outputs = model(mask_image_no_background)
  style_outputs = model(style_image)
  content_outputs = model(content_image)
  
  mask_outputs_idx=[]
  content_layers_idx=[]
  style_layers_idx=[]
  for i in range(len(layers_names)):
      if layers_names[i] in content_layers:
          content_layers_idx.append(i)
      if layers_names[i] in style_layers:
          style_layers_idx.append(i)
      if layers_names[i] in mask_layers:
          mask_outputs_idx.append(i)
  
  # Get the style and content feature representations from our model  
  style_features = [style_outputs[style_layers_idx[i]] for i in range(len(style_layers_idx))]
  content_features = [content_outputs[content_layers_idx[i]] for i in range(len(content_layers_idx))]
  mask_features = [mask_outputs[mask_outputs_idx[i]] for i in range(len(mask_outputs_idx))]

  
  
  return style_features, content_features,mask_features,style_layers_idx, content_layers_idx, mask_outputs_idx, content_masks, style_masks


    

def compute_loss(model, 
                 layers_names,
                 loss_weights, 
                 init_image,
                 content_features,
                 style_features,
                 style_layers_idx,
                 content_layers_idx,
                 content_masks,
                 style_masks,
                 Grid_image,
                 mask_features,
                 mask_idx,
                 mask_edges,
                 weightTone,
                 content_seg_path, 
                 style_seg_path):
  
  style_weight, content_weight, mask_weight, content_mask_weight,style_mask_weight = loss_weights
  
  
  model_outputs = model(init_image)
  
  style_output_features_var = [model_outputs[style_layers_idx[i]] for i in range(len(style_layers_idx)) ]
  content_output_features_var =[model_outputs[content_layers_idx[i]] for i in range(len(content_layers_idx))] 
  mask_output_features_var =[model_outputs[mask_idx[i]] for i in range(len(mask_idx))] 
  
     
  style_score = 0
  content_score = 0


  content_masks, style_masks = load_seg(content_seg_path, style_seg_path, [content_width, content_height], [style_width, style_height])
  
  
  ##################Grid Loss #############################
  init_image_gray = tf.image.rgb_to_grayscale(deprocess_img2(init_image))
  
    
  
  stp_score = get_Binary_Mask_loss(init_image_gray,Grid_image,content_masks[0])
  stp_score *= mask_weight
  
  

  ##################Style Losses #############################


  style_score_list, style_mask_score_list= segment_style_loss(model, layers_names, style_layers, mask_layers, style_features, style_output_features_var,mask_features, mask_output_features_var, content_masks, style_masks, float(1e2))
  style_score = 0.0
  for loss in style_score_list:
    style_score += loss
    
  style_mask_score = 0.0
  weight_per_style_mask_layer = 1.0 / float(num_mask_layers)
  for loss in style_mask_score_list:
    style_mask_score += weight_per_style_mask_layer*loss  
    
  ##################Content  Losses #############################
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features_var):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content,mask_edges)
  
  ##################Style Losses #############################
  content_mask_score = 0  
  weight_per_content_mask_layer = 1.0 / float(num_mask_layers)
  for target_content, comb_content in zip(mask_features, mask_output_features_var):
    content_mask_score += weight_per_content_mask_layer* get_content_loss(comb_content[0], target_content,1-mask_edges)
  
    
  ################### Tone Losss #######################
  gauss_kernel = gaussian_kernel(3,0.0,8.0)
    
  # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
  gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
  inputIm = tf.image.rgb_to_grayscale(deprocess_img2(init_image))
  # Convolve.
  Iblur= tf.nn.conv2d(inputIm, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
             
  Icont = deprocess_img2(content_image)
  
  LossTone = tf.reduce_mean(tf.square(Iblur-Icont))
  
  content_mask_score *= content_mask_weight
  style_mask_score *= style_mask_weight
  style_score *= style_weight
  content_score *= content_weight
  LossTone *= weightTone
  
  # Get total loss
  loss = style_score + content_score + stp_score + content_mask_score + style_mask_score + LossTone
  return loss, style_score, content_score, stp_score , content_output_features_var, style_output_features_var, style_mask_score, content_mask_score,LossTone


def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss


import IPython.display



def run_style_transfer(content_path, 
                       style_path,
                       mask_path,
                       num_iterations,
                       content_weight, 
                       style_weight,
                       mask_weight,
                       content_mask_weight,
                       style_mask_weight,
                       content_seg_path,
                       style_seg_path,
                       alpha,
                       weightTone): 
 
  global ite
  global display_interval 
  model, layers_names = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  
  style_features, content_features, mask_features, style_layers_idx, content_layers_idx, mask_idx, content_masks, style_masks = get_feature_representations(model, layers_names, content_layers, style_layers, mask_layers, content_path, style_path,content_seg_path,style_seg_path,mask_path)

  #gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  
  #init_image = load_init_noisy_image(256,256)
  
  
  init_image,edges,_ = init_mix_image(content_path,mask_path,alpha)
  
  edges = (edges+0.5)/1.5
  
  #init_image = load_and_process_img(content_path)
    
  init_image = tfe.Variable(init_image, dtype=tf.float32)
  
  
  
  _, Grid_image = load_mask3(mask_path,content_path,255/2-50)
  Grid_image=Grid_image/255
  # Create our optimizer
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

  # For displaying intermediate images 
  iter_count = 1
  
  # Store our best result
  best_loss, best_img = float('inf'), None
  
  loss_weights = (style_weight, content_weight, mask_weight,content_mask_weight,style_mask_weight)
  cfg = {
      'model': model,
      'layers_names':layers_names,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'content_features': content_features,
      'style_features': style_features,
      'content_layers_idx':content_layers_idx,
      'style_layers_idx':style_layers_idx,
      'content_masks':content_masks,
      'style_masks':style_masks,
      'Grid_image':Grid_image,
      'mask_features':mask_features,
      'mask_idx':mask_idx,
      'mask_edges':edges,
      'weightTone':weightTone,
      'content_seg_path':content_seg_path,
      'style_seg_path':style_seg_path
  }
    
  # For displaying
  num_rows = 2
  num_cols = 5
  #display_interval = 900
  display_interval = num_iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  imgs = []
  losses = []
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score, stp_score, content_output_features_var, style_output_features_var, style_mask_score, content_mask_score, ToneLoss = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    losses.append(best_loss)
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())
    
    ite = i      
    if i % display_interval== 0:
      start_time = time.time()
      """
      for j in range(len(style_output_features_var)):
          plotNNFilter(style_output_features_var[j],'Style Variable '+ str(j))
      for j in range(len(content_output_features_var)):
          plotNNFilter(content_output_features_var[j],'Content Variable '+ str(j))
      """
      # Use the .numpy() method to get the concrete numpy array
      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      imgs.append(plot_img)
      #IPython.display.clear_output(wait=True)
      #IPython.display.display_png(Image.fromarray(plot_img))
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'Mask loss: {:.4e}, '
            'style_mask_score: {:.4e},'
            'content_mask_score: {:.4e},'
            'Tone Loss: {:.4e},'
            'time: {:.4f}s'.format(loss, style_score, content_score,stp_score,style_mask_score, content_mask_score,ToneLoss, time.time() - start_time))
  print('Total time: {:.4f}s'.format(time.time() - global_start))
  #IPython.display.clear_output(wait=True)
  #plt.figure(figsize=(14,4))
  #for i,img in enumerate(imgs):
  #    plt.subplot(num_rows,num_cols,i+1)
  #    plt.imshow(img)
  #    plt.xticks([])
  #    plt.yticks([])
      
  return best_img, best_loss, imgs, losses  

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

#%%

content_layers = ['block2_conv2'] 
style_layers = ['block1_conv2','block3_conv2']
mask_layers = ['block1_conv1']    
Ratio_List =[0.7,0.8,1.1]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
num_mask_layers = len(mask_layers)

dirName='./Data/InputsDec21/OutputDec21v3/'
inputFolder='./Data/InputsDec21/'

def HedcutDrawings(args):
    dirName=args.data_folder+"/output/"
    inputFolder=args.data_folder
    global content_width
    global content_height
    #global mask_path
    global sizeSt
    #global content_path
    #global content_seg_path
    #global style_seg_path
    global style_width
    global style_height
    global content_image
    global best
    global content


    content_weight=5000
    style_weight=100
    #mask_weight=10000
    #mask_weight=0
    num_iterations=500
    alpha=0.8#Smaller more mask
    #content_mask_weight = 5000
    content_mask_weight = 100000
    style_mask_weight=10

    image_filesContent = sorted(glob('{}/*.pn*g'.format(inputFolder + '/Content/Grid')))
    image_filesStyles = sorted(glob('{}/*.pn*g'.format(inputFolder+ '/Style/Image')))
    

    
    for jj in range(len(image_filesContent)):
        for ll in range(len(image_filesStyles)):
            #replace("is", "was")
            content_path = image_filesContent[jj].replace("Grid","Image")
            style_path = image_filesStyles[ll]
            mask_path = image_filesContent[jj].replace("Image","Grid")
            content_seg_path = image_filesContent[jj].replace("Grid","Segmentation")
            style_seg_path = image_filesStyles[ll].replace("Image","Segmentation")
            
            for ii in range(len(Ratio_List)):
                Ratio = Ratio_List[ii]
                content_image = load_and_process_img(content_path)
                sizeSt = round(1/Ratio*content_image.shape[1])
                style_image = load_and_process_img_difSize(style_path,sizeSt)
                content_width, content_height = content_image.shape[2], content_image.shape[1]
                style_width, style_height = style_image.shape[2], style_image.shape[1]
                
                Ratio=content_image.shape[1]/style_image.shape[1]
                mask_weight=0
                
            

                
                content = load_img(content_path).astype('uint8')
                style = load_img(style_path).astype('uint8')
                
                
                # This is the line on main that executes the code  
              
    
                #style_mask_weight_List = [1e-10,5e-11,9e-11,1e-11,10e-11]
                weightTone = 0
                #style_mask_weight_List =[0]
                _, Grid_aux= load_mask3(mask_path,content_path,255/2-50)
                Grid_image = np.array(Grid_aux/255)
                _,edgesST,InitialImage= init_mix_image(content_path,mask_path,alpha)
                
                                 
                ite = 0
                best, best_loss , imgs,losses = run_style_transfer(content_path, 
                                                 style_path, mask_path, num_iterations, content_weight, style_weight,mask_weight, content_mask_weight, style_mask_weight, content_seg_path, style_seg_path,alpha,weightTone)
                segmentation_mask=load_img(content_seg_path).astype('uint8')
                content_outline=XDoG_Outline(np.squeeze(content))
                inner2 = XDoG_Dots(best)
                out=np.multiply(inner2.astype('float64'),content_outline.astype('float64'))
                out = (255/out.max()*out).astype('uint8')
                out[segmentation_mask[0,:,:,0]<35]  = 255
                out_Im = Image.fromarray(out,mode='L')    
                Filename=dirName+'C_%04d_S_%04d_R_%02d.png' % (jj,ll,ii)
                
                if not os.path.exists(dirName):
                  os.makedirs(dirName)
                out_Im.save(Filename)
                
                #timestamp = today.strftime("%A, %d %B %Y %I%M%p")
                sio.savemat(dirName+'C_%04d_S_%04d_R_%02d.png' % (jj,ll,ii) +'.mat', {'segmentation_mask':segmentation_mask,'num_iterations':num_iterations,'weightTone':weightTone,'Ratio':Ratio,'losses':losses,
                            'best':best,'imgs':imgs,'content_weight':content_weight,'style_weight':style_weight,'mask_weight':mask_weight,'alpha':alpha,'content_mask_weight':content_mask_weight,
                            'style_mask_weight':style_mask_weight,'style_layers':style_layers,'content_layers':content_layers,'mask_layers':mask_layers,'edgesST':edgesST,'style_height':style_height,
                            'content_height':content_height,'InitialImage':InitialImage,'content_path':content_path,'style_path':style_path,'mask_path':mask_path,'Grid_image':Grid_image,
                            'content_image':content,'style_image':style})
            plt.close('all')

if __name__ == '__main__':
    HedcutDrawings()
    
    

   