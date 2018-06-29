# -*- coding: utf-8 -*-
"""
Dinghan Shen

Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms (ACL 2018)
"""

import os
GPUID = 1
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
import sys
import scipy.io as sio
from math import floor
import pdb

from model import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx, tensors_key_in_file, prepare_data_for_emb

# import tempfile
# from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

class Options(object):
    def __init__(self):
        self.fix_emb = False
        self.reuse_w = True
        self.reuse_cnn = False
        self.reuse_discrimination = True  # reuse cnn for discrimination
        self.restore = True
        self.tanh = True  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 305
        self.n_words = None
        self.filter_shape = 5
        self.embed_size = 300
        self.lr = 3e-4
        self.layer = 3
        self.stride = [2, 2]  # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 128
        self.max_epochs = 1000
        self.n_gan = 500  # self.filter_size * 3
        self.L = 100
        self.drop_rate = 0.8
        self.encoder = 'concat'  # 'max' 'concat'

        self.part_data = False
        self.portion = 0.001   # 10%  1%  float(sys.argv[1])

        self.save_path = "./save/yahoo_emb"
        self.log_path = "./log"
        self.print_freq = 500
        self.valid_freq = 500

        self.discrimination = False
        self.dropout = 0.5
        self.H_dis = 300

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        # self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def emb_classifier(x, x_mask, y, dropout, opt):
    # print x.get_shape()  # batch L
    x_emb, W_emb = embedding(x, opt)  # batch L emb
    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
    x_emb = tf.nn.dropout(x_emb, dropout)   # batch L emb 1

    x_mask = tf.expand_dims(x_mask, axis=-1)
    x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb 1
    H_enc = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb 1
    H_enc = tf.squeeze(H_enc)  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, 3])  # batch 1
    H_enc_1 = H_enc / x_mask_sum  # batch emb

    H_enc_2 = tf.nn.max_pool(x_emb, [1, opt.maxlen, 1, 1], [1, 1, 1, 1], 'VALID')
    H_enc_2 = tf.squeeze(H_enc_2)

    H_enc = tf.concat([H_enc_1, H_enc_2], 1)

    H_enc = tf.squeeze(H_enc)
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=10, is_reuse=None)  # batch * 10
    prob = tf.nn.softmax(logits)

    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    train_op = layers.optimize_loss(
        loss,
        framework.get_global_step(),
        optimizer='Adam',
        learning_rate=opt.lr)

    return accuracy, loss, train_op, W_emb


def main():
    # global n_words
    # Prepare training and testing data
    loadpath = "./data/yahoo.p"
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]

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
        train_ind = np.random.choice(len(train), int(len(train)*opt.portion), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]

    try:
        params = np.load('./param_g.npz')
        if params['Wemb'].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            opt.W_emb = params['Wemb']
        else:
            print('Emb Dimension mismatch: param_g.npz:' + str(params['Wemb'].shape) + ' opt: ' + str(
                (opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen])
        x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen])
        keep_prob = tf.placeholder(tf.float32)
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, 10])
        accuracy_, loss_, train_op, W_emb_ = emb_classifier(x_, x_mask_, y_, keep_prob, opt)
        # merged = tf.summary.merge_all()

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

                t_vars = tf.trainable_variables()
                # print([var.name[:-2] for var in t_vars])
                save_keys = tensors_key_in_file(opt.save_path)
                # print(save_keys.keys())
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
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
                kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
                for _, train_index in kf:
                    uidx += 1
                    sents = [train[t] for t in train_index]
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    x_labels = x_labels.reshape((len(x_labels), 10))

                    x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)

                    _, loss = sess.run([train_op, loss_], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.drop_rate})

                    if uidx % opt.valid_freq == 0:
                        train_correct = 0.0
                        kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
                        for _, train_index in kf_train:
                            train_sents = [train[t] for t in train_index]
                            train_labels = [train_lab[t] for t in train_index]
                            train_labels = np.array(train_labels)
                            train_labels = train_labels.reshape((len(train_labels), 10))
                            x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)  # Batch L

                            train_accuracy = sess.run(accuracy_, feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, y_: train_labels, keep_prob: 1.0})

                            train_correct += train_accuracy * len(train_index)

                        train_accuracy = train_correct / 500

                        print("Iteration %d: Training loss %f " % (uidx, loss))
                        print("Train accuracy %f " % train_accuracy)

                        val_correct = 0.0
                        kf_val = get_minibatches_idx(20000, opt.batch_size, shuffle=True)
                        for _, val_index in kf_val:
                            val_sents = [val[t] for t in val_index]
                            val_labels = [val_lab[t] for t in val_index]
                            val_labels = np.array(val_labels)
                            val_labels = val_labels.reshape((len(val_labels), 10))
                            x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)

                            val_accuracy = sess.run(accuracy_, feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask,
                                                                          y_: val_labels, keep_prob: 1.0})

                            val_correct += val_accuracy * len(val_index)

                        val_accuracy = val_correct / 20000
                        print("Validation accuracy %f " % val_accuracy)

                        if val_accuracy > max_val_accuracy:
                            max_val_accuracy = val_accuracy

                            test_correct = 0.0
                            kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=True)
                            for _, test_index in kf_test:
                                test_sents = [test[t] for t in test_index]
                                test_labels = [test_lab[t] for t in test_index]
                                test_labels = np.array(test_labels)
                                test_labels = test_labels.reshape((len(test_labels), 10))
                                x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)

                                test_accuracy = sess.run(accuracy_,
                                                         feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,
                                                                    y_: test_labels, keep_prob: 1.0})

                                test_correct += test_accuracy * len(test_index)

                            test_accuracy = test_correct / len(test)

                            print("Test accuracy %f " % test_accuracy)

                            max_test_accuracy = test_accuracy

                print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))

                emb = sess.run(W_emb_, feed_dict={x_: x_test_batch})

                cPickle.dump([emb], open("yahoo_emb_max_300.p", "wb"))

            print("Max Test accuracy %f " % max_test_accuracy)

        except KeyboardInterrupt:
            # print 'Training interupted'
            print('Training interupted')
            print("Max Test accuracy %f " % max_test_accuracy)

if __name__ == '__main__':
    main()

