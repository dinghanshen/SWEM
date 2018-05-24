#! /usr/bin/env python

import os, sys
import skimage
import skimage.io
import skimage.transform
import numpy as np
from IPython import embed
import tensorflow as tf
from keras.layers import Input
import keras.backend as K
from keras.models import Sequential, Model

#pack_root = os.path.join(os.path.dirname(__file__), '..', '..')
#sys.path.insert(1, pack_root)

# extended Keras layers
from keras_layers import *


def sample_box(proposed_box, target_box,high_thresh, low_thresh,batch_size):
    """ Compute Box IOU and sample positive/negative boxes and targe boxes
        Input:
            - proposed_box: tensor, all of the proposed boxes from RPN model.
            - target_box: tensor, groudtruth box from input dataset.
            - high_thresh: float, iou threshold to pick positive samples.
            - low_thresh: float, iou threshold to pick negative samples.
            - batch_sizes: output sample size.
        Output:
            - packed_pos_samples: tensor, packed with pos_samples and neg_samples.
            - negative_samples: tensor.
    """
    # NOTE: this function should goes to model_builder.py later.
    out_iou = BoxIoU()([proposed_box, target_box])
    
    sample_idx = BoxSamplerPosNeg(high_thresh, low_thresh, batch_size)(out_iou)
    ## NOTE: pos_samples is packed with pos_samples and tar_samples. Do NOT unpack here, 
    ##       otherwise keras cannot recognize the tensor size. 
    #packed_pos_samples = BoxSamplerPositive(high_thresh, batch_size)(
    #        [proposed_box, target_box,out_iou])
    #neg_samples = BoxSamplerNegative(low_thresh, batch_size)([proposed_box, out_iou])

    model = Model(input=[proposed_box, target_box], output=[
                         sample_idx])
    return model


def test_box_sampling():
    print 'Test box sampling module ...'
    # build keras model graph
    in_box1 = Input(batch_shape=(1,3, 4)) # proposed box
    in_box2 = Input(batch_shape=(1,2, 4)) # target box
    model = sample_box(in_box1, in_box2, 0.1, 0.1, 2)


    # create testing input values
    in_box1_val = np.array([[20., 10., 5., 5.],
                         [80., 10., 5., 20.],
                         [80., 80., 10., 5.]])
    in_box1_val = np.tile(in_box1_val, (1,1,1))
    in_box2_val = np.array([[20., 10., 20., 10.],
                         [80., 80., 10., 10.]])
    in_box2_val = np.tile(in_box2_val, (1,1,1))

    # run graph
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    out_vals = sess.run(model.output, feed_dict={
        model.input[0]: in_box1_val,
        model.input[1]: in_box2_val})
    
    print 'box sampling OK!'
    embed()

def test_boxiou():
    print 'Test Box IOU layer...'
    # build keras model graph
    in_box1 = Input(batch_shape=(1,3, 4)) # proposed box
    in_box2 = Input(batch_shape=(1,2, 4)) # target box
    out_iou = BoxIoU()([in_box1, in_box2])
    model = Model(input=[in_box1, in_box2], output=out_iou)

    # create testing input values
    in_box1_val = np.array([[20., 10., 5., 5.],
                         [80., 10., 5., 20.],
                         [80., 80., 10., 5.]])
    in_box1_val = np.tile(in_box1_val, (1,1,1))
    in_box2_val = np.array([[20., 10., 20., 10.],
                         [80., 80., 10., 10.]])
    in_box2_val = np.tile(in_box2_val, (1,1,1))

    # run graph
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    out_iou_val = sess.run(model.output, feed_dict={
        model.input[0]: in_box1_val,
        model.input[1]: in_box2_val})
    
    print 'Box IOU OK!'
    print out_iou_val

def test_selectpos():
    print 'Test SelectPosMakeTheta layer...'
    in_sample_index = Input(batch_shape=(5,3))  # sample index
    in_box_coords = Input(batch_shape=(6,4))

    out_theta = SelectPosMakeTheta(64,64)([in_sample_index, in_box_coords])
    model = Model(input=[in_sample_index, in_box_coords], output = out_theta)

    # create some data
    sample_index = np.array([[1, 2, 1],
                            [1, 0, 3],
                            [1, 4, 2],
                            [-1,1, -1],
                            [-1,3, -1]])
    box_coords = np.array([[0., 0., 12., 14.],
                        [1., 2., 15., 15.],
                        [1.5, 2., 4., 10.],
                        [5., 8., 4., 10.],
                        [5.5, 3., 6., 8.],
                        [3., 4., 9., 9.]])
    # run graph
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    out_theta_val = sess.run(model.output, feed_dict={
        model.input[0]: sample_index,
        model.input[1]: box_coords})
    
    print 'SelectPosMakeTheta works!'
    print out_theta_val
        
def test_tile():
    in_x = Input(batch_shape = (1,13,13,5))
    in_y = Input(batch_shape = (12,6))

    out_x = TileTensorLike()([in_x, in_y])
    model = Model(input=[in_x,in_y], output=out_x)

    in_x_val = np.random.rand(1,13,13,5)
    in_y_val = np.random.rand(12,6)

    # run graph
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    out_x_val = sess.run(model.output, feed_dict={
        model.input[0]: in_x_val,
        model.input[1]: in_y_val})
    
    print 'Tile works!'
    print out_x_val.shape

if __name__ == '__main__':
    #test_boxiou()
    #test_box_sampling()
    #test_selectpos()
    test_tile()


