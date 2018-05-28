# -*- coding: utf-8 -*-
"""
Dinghan Shen

Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms (ACL 2018)
"""


import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib import metrics
from tensorflow.contrib import slim
# from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder, sequence_loss, embedding_rnn_seq2seq, \
    embedding_tied_rnn_seq2seq
import pdb
import copy
from utils import normalizing
from tensorflow.python.ops import nn_ops, math_ops
import numpy as np


def embedding(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    #  b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=True)
            #pdb.set_trace()
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
            # tf.stop_gradient(W)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    # W_norm = normalizing(W, 1)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W


def aver_emb_encoder(x_emb, x_mask):
    """ compute the average over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    H_enc_0 = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc_0, [1, 3])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc


def max_emb_encoder(x_emb, x_mask, opt):
    """ compute the max over every dimension of word embeddings """
    x_mask_1 = tf.expand_dims(x_mask, axis=-1)
    x_mask_1 = tf.expand_dims(x_mask_1, axis=-1)
    H_enc = tf.nn.max_pool(tf.multiply(x_emb, x_mask_1), [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    H_enc = tf.squeeze(H_enc)

    return H_enc

def concat_emb_encoder(x_emb, x_mask, opt):
    """ concat both the average and max over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    H_enc = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc, [1, 3])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1

    H_enc_1 = H_enc / x_mask_sum  # batch emb

    H_enc_2 = tf.nn.max_pool(x_emb, [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    H_enc_2 = tf.squeeze(H_enc_2, [1, 3])

    H_enc = tf.concat([H_enc_1, H_enc_2], 1)

    return H_enc


def discriminator_1layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    return H_dis


def discriminator_0layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    logits = layers.linear(tf.nn.dropout(H, keep_prob=dropout), num_outputs=num_outputs, biases_initializer=biasInit,
                           scope=prefix + 'dis', reuse=is_reuse)
    return logits


def linear_layer(x, output_dim):
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0 / (input_dim + output_dim))
    W = tf.get_variable("W", [input_dim, output_dim], scope=prefix + '_W',
                        initializer=tf.random_uniform_initializer(minval=-thres, maxval=thres))
    b = tf.get_variable("b", [output_dim], scope=prefix + '_b', initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b


def discriminator_2layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    # H = tf.squeeze(H, [1,2])
    # pdb.set_trace()
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
    return logits

def discriminator_3layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    # H = tf.squeeze(H, [1,2])
    # pdb.set_trace()
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    H_dis = layers.fully_connected(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_2',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_3', reuse=is_reuse)
    return logits

def discriminator_res(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    # H = tf.squeeze(H, [1,2])
    # pdb.set_trace()
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis_0 = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    H_dis_0n = tf.nn.relu(H_dis_0)                               
    H_dis_1 = layers.fully_connected(tf.nn.dropout(H_dis_0n, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_2',
                                   reuse=is_reuse)
    H_dis_1n = tf.nn.relu(H_dis_1) + H_dis_0
    H_dis_2 = layers.fully_connected(tf.nn.dropout(H_dis_1n, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_3',
                                   reuse=is_reuse)
    H_dis_2n = tf.nn.relu(H_dis_2) + H_dis_1
    H_dis_3 = layers.fully_connected(tf.nn.dropout(H_dis_2n, keep_prob=dropout), num_outputs=opt.embed_size,
                                   biases_initializer=biasInit, activation_fn=None, scope=prefix + 'dis_4',
                                   reuse=is_reuse)

    logits = layers.linear(tf.nn.dropout(H_dis_3, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_10', reuse=is_reuse)
    return logits
