from os import listdir
from os.path import isfile, join
import sys
import os 
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import argparse
import tensorflow as tf

def get_metrics(d,d_gt,mask):
    d = tf.convert_to_tensor(d, dtype = tf.float32)
    d_gt = tf.convert_to_tensor(d_gt, dtype = tf.float32)
    mask = tf.convert_to_tensor(mask, dtype = tf.bool)

    d = tf.boolean_mask(d, mask)
    d_gt = tf.boolean_mask(d_gt, mask)

    d = tf.reshape(d,[-1])
    d_gt = tf.reshape(d_gt,[-1])
  
    #AbsDiff(d,d_gt),
    e_vec = tf.convert_to_tensor( [AbsRel(d,d_gt),  SqRel(d,d_gt), RMSE(d,d_gt), log_scale_inv_RMSE(d, d_gt), accuracy_under_thres(d, d_gt),  accuracy_under_thres2(d, d_gt), accuracy_under_thres3(d, d_gt)], dtype = tf.float32)
    
    return e_vec

def AbsRel(d,d_gt):
    e = tf.cast(1 / tf.math.reduce_prod(tf.shape(d)), dtype = tf.float32)
    e = e*tf.reduce_sum(tf.math.divide(tf.math.abs(d-d_gt), d_gt))
    return e

def AbsDiff(d,d_gt):
    e = tf.cast(1 / tf.math.reduce_prod(tf.shape(d)), dtype = tf.float32)
    e = e * tf.reduce_sum(tf.math.abs(d-d_gt))
    return e

def SqRel(d,d_gt):
    e = tf.cast(1 / tf.math.reduce_prod(tf.shape(d)), dtype = tf.float32)
    e = e* tf.reduce_sum(tf.math.divide(tf.math.pow(d-d_gt, 2), d_gt))
    return e

def RMSE(d,d_gt):
    e = tf.cast(1 / tf.math.reduce_prod(tf.shape(d)), dtype = tf.float32)
    e = e * tf.reduce_sum(tf.math.pow(d-d_gt, 2))
    e = tf.math.sqrt(e)
    return e

def log_scale_inv_RMSE(d, d_gt):
    EPS = 1e-16

    alpha = tf.cast(1/(2*tf.math.reduce_prod(tf.shape(d))), dtype = tf.float32)
    alpha = alpha * tf.reduce_sum(tf.math.log(d_gt)-tf.math.log(d+EPS)) 
    e = tf.cast(1/(2*tf.math.reduce_prod(tf.shape(d))) , dtype = tf.float32)
    
    e = e * tf.reduce_sum(tf.math.pow(tf.math.log(d+EPS)-tf.math.log(d_gt)+alpha,2))
    return e

def accuracy_under_thres(d, d_gt):
    thres = 1.1
    e = tf.reduce_sum(tf.cast(tf.math.maximum(d/d_gt, d_gt/d) < thres, dtype = tf.float32))/tf.cast(tf.reduce_prod(tf.shape(d)), dtype = tf.float32) * 100
    return e

def accuracy_under_thres2(d, d_gt):
    thres = 1.25
    e = tf.reduce_sum(tf.cast(tf.math.maximum(d/d_gt, d_gt/d) < thres, dtype = tf.float32))/tf.cast(tf.reduce_prod(tf.shape(d)), dtype = tf.float32) * 100
    return e

def accuracy_under_thres3(d, d_gt):
    thres = 1.25**2
    e = tf.reduce_sum(tf.cast(tf.math.maximum(d/d_gt, d_gt/d) < thres, dtype = tf.float32))/tf.cast(tf.reduce_prod(tf.shape(d)), dtype = tf.float32) * 100
    return e

