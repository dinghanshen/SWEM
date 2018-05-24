import numpy as np
from IPython import embed
import pdb

import tensorflow as tf
from keras.layers import Dense, Flatten, Input
import keras.backend as K

from general.tfutil import hist_summaries_traintest, scalar_summaries_traintest

from keras_ext.engine import ExtendedModel
from keras_ext.layers import RProjDense, RProjConv2D, RProjBatchNormalization
from keras_ext.rproj_layers_util import OffsetCreatorDenseProj, OffsetCreatorSparseProj, ThetaPrime

from keras.regularizers import l2

def make_and_add_losses(model, input_labels, l2=0):
    '''Add classification and L2 losses'''

    with tf.name_scope('losses') as scope:
        prob = tf.nn.softmax(model.v.logits, name='prob')
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.v.logits, labels=input_labels, name='cross_ent')
        loss_cross_ent = tf.reduce_mean(cross_ent, name='loss_cross_ent')
        model.add_trackable('loss_cross_ent', loss_cross_ent)
        class_prediction = tf.argmax(prob, 1)
        #class_true = tf.argmax(in_y, 1)    # convert 1-hot to index

        prediction_correct = tf.equal(class_prediction, input_labels, name='prediction_correct')
        accuracy = tf.reduce_mean(tf.to_float(prediction_correct), name='accuracy')
        model.add_trackable('accuracy', accuracy)
        hist_summaries_traintest(prob, cross_ent)
        #scalar_summaries_traintest(loss_cross_ent, loss_spring, loss, accuracy)
        scalar_summaries_traintest(accuracy)

        model.add_loss_reg()
        if 'loss_reg' in model.v:
            loss = tf.add_n((
                model.v.loss_cross_ent,
                model.v.loss_reg,
            ), name='loss')
        else:
            loss = model.v.loss_cross_ent
        model.add_trackable('loss', loss)

    nontrackable_fields = ['prob', 'cross_ent', 'class_prediction', 'prediction_correct']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])


def build_model_fc_dir(state_dim, num_actions, weight_decay=0, depth=2, width=100, shift_in=None):
    state_shape = (1, state_dim)
    dtype = 'float32'

    with tf.name_scope('inputs'):
        input_state = Input(shape=state_shape, dtype=dtype)

    with tf.name_scope('net') as scope:
        xx = input_state
        xx = Flatten()(xx)
        for _ in range(depth):
            xx = Dense(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = Dense(num_actions, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_state, output=logits)

    nontrackable_fields = ['input_state', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    return model



def build_model_fourier(state_dim, num_actions, weight_decay=0, depth=2, width=100, shift_in=None):
    state_shape = (1, state_dim)
    dtype = 'float32'

    with tf.name_scope('inputs'):
        input_state = Input(shape=state_shape, dtype=dtype)

    with tf.name_scope('net') as scope:
        xx = input_state
        xx = Flatten()(xx)
        logits = Dense(num_actions, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_state, output=logits)

    nontrackable_fields = ['input_state', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    return model




def build_model_fc(state_dim, num_actions, weight_decay=0, vsize=100, depth=2, width=100, shift_in=None, proj_type='sparse'):
    state_shape = (1, state_dim)
    dtype = 'float32'

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        input_state = Input(shape=state_shape, dtype=dtype)

    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)

        xx = input_state
        xx = Flatten()(xx)
        for _ in range(depth):
            xx = RProjDense(offset_creator_class, vv, width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = RProjDense(offset_creator_class, vv, num_actions, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_state, output=logits)

        model.add_extra_trainable_weight(vv.var)

    nontrackable_fields = ['input_state', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    return model

