# -*- coding: utf-8 -*-
"""
Dinghan Shen

Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms (ACL 2018)
"""

import os
GPUID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
# from tensorflow.contrib import metrics
# from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
# from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb

from model import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save\
    , _clip_gradients_seperate_norm, tensors_key_in_file, prepare_data_for_emb

# import tempfile
# from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

class Options(object):
    def __init__(self):
        self.fix_emb = True
        #self.relu_w = True
        #self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = True  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = None  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 69
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 2  # filtersize multiplier
        self.embed_size = 300
        self.lr = 3e-4
        self.layer = 3
        self.stride = [2, 2, 2]  # for two layer cnn/deconv, use self.stride[0]
        self.batch_size = 307  # 9824
        self.max_epochs = 5000
        self.n_gan = 500  # self.filter_size * 3
        self.L = 100
        self.encoder = 'max'  # 'max' 'concat'
        self.combine_enc = 'mix'
        self.category = 3  # '1' for binary

        self.optimizer = 'RMSProp'  # tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.dropout_ratio = 0.8

        # self.save_path = "./save/snli_emb_10"
        self.save_path = "./save/snli_emb_10"
        self.log_path = "./log"
        self.valid_freq = 100

        # partially use labeled data
        self.part_data = False
        self.portion = 0.01  # 10%  1%

        self.discrimination = False
        self.H_dis = 300

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape) / self.stride[2]) + 1)
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def auto_encoder(x_1, x_2, x_mask_1, x_mask_2, y, dropout, opt):
    x_1_emb, W_emb = embedding(x_1, opt)  # batch L emb
    x_2_emb = tf.nn.embedding_lookup(W_emb, x_2)

    x_1_emb = tf.nn.dropout(x_1_emb, dropout)  # batch L emb
    x_2_emb = tf.nn.dropout(x_2_emb, dropout)  # batch L emb

    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    x_1_emb = layers.fully_connected(tf.squeeze(x_1_emb), num_outputs=opt.embed_size, biases_initializer=biasInit, activation_fn=tf.nn.relu, scope='trans', reuse=None)  # batch L emb
    x_2_emb = layers.fully_connected(tf.squeeze(x_2_emb), num_outputs=opt.embed_size, biases_initializer=biasInit, activation_fn=tf.nn.relu, scope='trans', reuse=True)

    x_1_emb = tf.expand_dims(x_1_emb, 3)  # batch L emb 1
    x_2_emb = tf.expand_dims(x_2_emb, 3)

    if opt.encoder == 'aver':
        H_enc_1 = aver_emb_encoder(x_1_emb, x_mask_1)
        H_enc_2 = aver_emb_encoder(x_2_emb, x_mask_2)

    elif opt.encoder == 'max':
        H_enc_1 = max_emb_encoder(x_1_emb, x_mask_1, opt)
        H_enc_2 = max_emb_encoder(x_2_emb, x_mask_2, opt)

    elif opt.encoder == 'concat':
        H_enc_1 = concat_emb_encoder(x_1_emb, x_mask_1, opt)
        H_enc_2 = concat_emb_encoder(x_2_emb, x_mask_2, opt)

    # discriminative loss term
    if opt.combine_enc == 'mult':
        H_enc = tf.multiply(H_enc_1, H_enc_2)  # batch * n_gan

    if opt.combine_enc == 'concat':
        H_enc = tf.concat([H_enc_1, H_enc_2], 1)

    if opt.combine_enc == 'sub':
        H_enc = tf.subtract(H_enc_1, H_enc_2)

    if opt.combine_enc == 'mix':
        H_1 = tf.multiply(H_enc_1, H_enc_2)
        H_2 = tf.concat([H_enc_1, H_enc_2], 1)
        H_3 = tf.subtract(H_enc_1, H_enc_2)
        H_enc = tf.concat([H_1, H_2, H_3], 1)

    # calculate the accuracy
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.category, is_reuse=None)
    prob = tf.nn.softmax(logits)

    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    train_op = layers.optimize_loss(
        loss,
        framework.get_global_step(),
        optimizer='Adam',
        # variables=d_vars,
        learning_rate=opt.lr)

    return accuracy, loss, train_op, W_emb

def main():
    loadpath = "./data/snli.p"
    x = cPickle.load(open(loadpath, "rb"))

    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[4], x[5]

    train_q, train_a, train_lab = train[0], train[1], train[2]
    val_q, val_a, val_lab = val[0], val[1], val[2]
    test_q, test_a, test_lab = test[0], test[1], test[2]

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')

    opt = Options()
    opt.n_words = len(ixtoword)

    del x

    print(dict(opt))
    print('Total words: %d' % opt.n_words)

    if opt.part_data:
        np.random.seed(123)
        train_ind = np.random.choice(len(train_q), int(len(train_q)*opt.portion), replace=False)
        train_q = [train_q[t] for t in train_ind]
        train_a = [train_a[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]

    try:
        params = np.load('./data/snli_emb.p')
        if params[0].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            #pdb.set_trace()
            opt.W_emb = np.array(params[0], dtype='float32')
        else:
            print('Emb Dimension mismatch: param_g.npz:' + str(params[0].shape) + ' opt: ' + str(
                (opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_1_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen])
        x_2_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen])
        x_mask_1_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen])
        x_mask_2_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen])
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.category])
        keep_prob = tf.placeholder(tf.float32)
        accuracy_, loss_, train_op_, W_emb_ = auto_encoder(x_1_, x_2_, x_mask_1_, x_mask_2_, y_, keep_prob, opt)
        merged = tf.summary.merge_all()

    uidx = 0
    max_val_accuracy = 0.
    max_test_accuracy = 0.
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                #pdb.set_trace()
                t_vars = tf.trainable_variables()
                # print([var.name[:-2] for var in t_vars])
                save_keys = tensors_key_in_file(opt.save_path)

                # pdb.set_trace()
                # print(save_keys.keys())
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
                #pdb.set_trace()

                # only restore variables with correct shape
                ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

                loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
                loader.restore(sess, opt.save_path)

                print("Loading variables from '%s'." % opt.save_path)
                print("Loaded variables:" + str(ss))

            except:
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(opt.max_epochs):
                print("Starting epoch %d" % epoch)
                kf = get_minibatches_idx(len(train_q), opt.batch_size, shuffle=True)
                for _, train_index in kf:

                    uidx += 1
                    sents_1 = [train_q[t] for t in train_index]
                    sents_2 = [train_a[t] for t in train_index]
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    x_labels = x_labels.reshape((len(x_labels), opt.category))

                    x_batch_1, x_batch_mask_1 = prepare_data_for_emb(sents_1, opt)
                    x_batch_2, x_batch_mask_2 = prepare_data_for_emb(sents_2, opt)

                    _, loss = sess.run([train_op_, loss_], feed_dict={x_1_: x_batch_1, x_2_: x_batch_2,
                                       x_mask_1_: x_batch_mask_1, x_mask_2_: x_batch_mask_2, y_: x_labels, keep_prob: opt.dropout_ratio})

                    if uidx % opt.valid_freq == 0:

                        train_correct = 0.0
                        kf_train = get_minibatches_idx(3070, opt.batch_size, shuffle=True)
                        for _, train_index in kf_train:
                            train_sents_1 = [train_q[t] for t in train_index]
                            train_sents_2 = [train_a[t] for t in train_index]
                            train_labels = [train_lab[t] for t in train_index]
                            train_labels = np.array(train_labels)
                            train_labels = train_labels.reshape((len(train_labels), opt.category))
                            x_train_batch_1, x_train_mask_1 = prepare_data_for_emb(train_sents_1, opt)
                            x_train_batch_2, x_train_mask_2 = prepare_data_for_emb(train_sents_2, opt)

                            train_accuracy = sess.run(accuracy_,
                                                      feed_dict={x_1_: x_train_batch_1, x_2_: x_train_batch_2, x_mask_1_: x_train_mask_1, x_mask_2_: x_train_mask_2,
                                                                 y_: train_labels, keep_prob: 1.0})

                            train_correct += train_accuracy * len(train_index)

                        train_accuracy = train_correct / 3070

                        # print("Iteration %d: Training loss %f, dis loss %f, rec loss %f" % (uidx,
                        #                                                                     loss, dis_loss, rec_loss))
                        print("Train accuracy %f " % train_accuracy)

                        val_correct = 0.0
                        is_train = True
                        kf_val = get_minibatches_idx(len(val_q), opt.batch_size, shuffle=True)
                        for _, val_index in kf_val:
                            val_sents_1 = [val_q[t] for t in val_index]
                            val_sents_2 = [val_a[t] for t in val_index]
                            val_labels = [val_lab[t] for t in val_index]
                            val_labels = np.array(val_labels)
                            val_labels = val_labels.reshape((len(val_labels), opt.category))
                            x_val_batch_1, x_val_mask_1 = prepare_data_for_emb(val_sents_1, opt)
                            x_val_batch_2, x_val_mask_2 = prepare_data_for_emb(val_sents_2, opt)

                            val_accuracy = sess.run(accuracy_, feed_dict={x_1_: x_val_batch_1, x_2_: x_val_batch_2,
                                                                          x_mask_1_: x_val_mask_1, x_mask_2_: x_val_mask_2, y_: val_labels, keep_prob: 1.0})

                            val_correct += val_accuracy * len(val_index)

                        val_accuracy = val_correct / len(val_q)

                        print("Validation accuracy %f " % val_accuracy)

                        if val_accuracy > max_val_accuracy:
                            max_val_accuracy = val_accuracy

                            test_correct = 0.0
                            kf_test = get_minibatches_idx(len(test_q), opt.batch_size, shuffle=True)
                            for _, test_index in kf_test:
                                test_sents_1 = [test_q[t] for t in test_index]
                                test_sents_2 = [test_a[t] for t in test_index]
                                test_labels = [test_lab[t] for t in test_index]
                                test_labels = np.array(test_labels)
                                test_labels = test_labels.reshape((len(test_labels), opt.category))
                                x_test_batch_1, x_test_mask_1 = prepare_data_for_emb(test_sents_1, opt)
                                x_test_batch_2, x_test_mask_2 = prepare_data_for_emb(test_sents_2, opt)

                                test_accuracy = sess.run(accuracy_, feed_dict={x_1_: x_test_batch_1, x_2_: x_test_batch_2,
                                                                               x_mask_1_: x_test_mask_1, x_mask_2_: x_test_mask_2,
                                                                               y_: test_labels, keep_prob: 1.0})

                                test_correct += test_accuracy * len(test_index)

                            test_accuracy = test_correct / len(test_q)

                            print("Test accuracy %f " % test_accuracy)

                            max_test_accuracy = test_accuracy

                print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))

            print("Max Test accuracy %f " % max_test_accuracy)

        except KeyboardInterrupt:
            print('Training interupted')
            print("Max Test accuracy %f " % max_test_accuracy)

if __name__ == '__main__':
    main()