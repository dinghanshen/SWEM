import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Flatten, Input, Lambda

from general.tfutil import tf_assert_gpu, hist_summaries_traintest


########################
# General Keras helpers
########################

def make_image_input_preproc(im_dims, dtype='float32', flatten_in=False, shift_in=None, name=None):
    '''Make an input for images and (optionally preprocess). Returns
    both the Input layer (which should be used as Model input) and the
    preproc version (which should be passed to the first layer of the
    model). If no preprocessing is done, the Input layer and preproc
    will be the same.
    '''
    
    assert isinstance(im_dims, tuple) and len(im_dims) == 3, 'should be tuple of 3 dims (0,1,c)'
    assert dtype in ('float32', 'uint8'), 'unknown dtype'

    input_images = Input(shape=im_dims, dtype=dtype, name=name)
    preproc_images = input_images
    if dtype == 'uint8':
        preproc_images = Lambda(lambda x: K.cast(x, 'float32'))(preproc_images)
    if shift_in is not None:
        print 'subtracting from each input:', shift_in
        preproc_images = Lambda(lambda x: x - shift_in)(preproc_images)
    if flatten_in:
        preproc_images = Flatten()(preproc_images)
    return input_images, preproc_images


def make_classlabel_input(n_label_vals):
    return Input(batch_shape=(None,), dtype='int64')


def setup_session_and_seeds(seed, assert_gpu=True, mem_fraction=None):
    '''Start TF and register session with Keras'''

    # Use InteractiveSession instead of Session so the default session will be set
    if mem_fraction is not None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.InteractiveSession()
    K.set_session(sess)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    print 'Set numpy and tensorflow random seeds to: %s' % repr(seed)
    print 'My PID is %d' % os.getpid()
    if assert_gpu:
        tf_assert_gpu(sess)
    return sess


def add_act_summaries(model, quiet=False):
    tensors = []
    if not quiet:
        print '\nActivations:'
    for layer in model.layers:
        for node in layer._inbound_nodes:
            for tensor in node.output_tensors:
                tensors.append(tensor)
    tdict = {tt.name: tt for tt in set(tensors)}
    for tname in sorted(tdict.keys()):
        hist_summaries_traintest(tdict[tname], name=tname + '__act')
        if not quiet:
            print '  ', tname, tdict[tname]


def get_model_tensors(model, with_layers_nodes=False):
    tensor_set = set()
    tensor_list = []
    layer_list = []
    node_list = []
    for layer in model.layers:
        for node in layer._inbound_nodes:
            for tensor in node.output_tensors:
                if tensor not in tensor_set:
                    # Make a list with deteministic order, but check membership using a fast set
                    tensor_set.add(tensor)
                    tensor_list.append(tensor)
                    layer_list.append(layer)
                    node_list.append(node)
    if with_layers_nodes:
        return tensor_list, layer_list, node_list
    else:
        return tensor_list


def warn_misaligned_shapes(model):
    printed = False
    tlns = get_model_tensors(model, with_layers_nodes=True)
    for tln in zip(tlns[0], tlns[1], tlns[2]):
        tensor, layer, node = tln
        tf_shape = tuple(tensor.get_shape().as_list())
        try:
            keras_shape = tensor._keras_shape
        except AttributeError:
            continue
        if tf_shape != keras_shape:
            if not printed:
                print '\nWarning: found the following tensor shape mismatches, may indicate problems.'
                print '   %-40s %-22s %-22s' % ('LAYER NAME', '', '')
                print '   %-40s %-22s %-22s' % ('TENSOR NAME', 'KERAS SHAPE', 'TF SHAPE')
                printed = True
            print '   %-40s %-22s %-22s' % (layer.name, '', '')
            print '   %-40s %-22s %-22s' % (tensor.name, keras_shape, tf_shape)


def full_static_shape(var):
    '''Returns the most fully-specified static shape possible for var (at
    graph build time, not run time). Uses information in
    var.get_shape() as well as var._keras_shape. Raises an Exception
    if the two shapes are incompatible with each other.
    '''

    try:
        tf_shape = [val.__int__() for val in var.get_shape()]
    except ValueError:
        raise Exception('Unclear why this would ever be encountered. If it pops up, debug here.')
            
    if not hasattr(var, '_keras_shape'):
        return tf_shape

    k_shape = var._keras_shape
    assert len(tf_shape) == len(k_shape), 'Shape lengths different; this probably should not occur'

    shape = []
    for tt, kk in zip(tf_shape, k_shape):
        if tt == kk:
            shape.append(tt)
        else:
            if tt is None:
                shape.append(kk)
            elif kk is None:
                shape.append(tt)
            else:
                raise Exception('tf shape and Keras shape are contradictory: %s vs %s' % (tf_shape, k_shape))
    return shape
