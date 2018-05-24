#! /usr/bin/env python

import sys
import os
import gzip
import cPickle as pickle
import numpy as np
import h5py
from IPython import embed
import colorama
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
import cPickle
import pdb
from keras.datasets import imdb
from load_glove_embeddings import load_glove_embeddings

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(1, lab_root)

from general.util import tic, toc, tic2, toc2, tic3, toc3, mkdir_p, WithTimer
from general.image_preproc import ImagePreproc
from general.stats_buddy import StatsBuddy
from general.tfutil import (get_collection_intersection_summary,
                           log_scalars, sess_run_dict, 
                           summarize_weights, summarize_opt,
                           tf_assert_all_init, 
                           tf_get_uninitialized_variables, 
                           add_grad_summaries)
from keras_ext.util import setup_session_and_seeds, warn_misaligned_shapes
from model_builders import (build_model_text_swem_dir,
                            build_model_text_swem,
                            build_model_text_swem_fastfood,
                            build_model_text_cnn_dir,
                            build_model_text_cnn,
                            build_model_text_cnn_fastfood)
from standard_parser import make_standard_parser


arch_choices_direct = ['swem_dir','cnn_dir']
arch_choices_projected = ['swem','cnn']
dataset_choices = ['agnews','yelp']
arch_choices = arch_choices_direct + arch_choices_projected

class LRStepper(object):
    def __init__(self, lr_init, lr_ratio, lr_epochs, lr_steps):
        self.lr_init = lr_init
        self.lr_ratio = lr_ratio
        self.lr_epochs = lr_epochs
        self.lr_steps = lr_steps
        self.last_printed = None

    def lr(self, buddy):
        if self.lr_ratio == 0 or self.lr_epochs == 0:
            return self.lr_init
        ret = self.lr_init * self.lr_ratio ** int(min(buddy.epoch / self.lr_epochs, self.lr_steps))
        if ret != self.last_printed:
            print 'At epoch %d setting LR to %g' % (buddy.epoch, ret)
            self.last_printed = ret
        return ret

def main():
    parser = make_standard_parser('Random Projection Experiments.', arch_choices=arch_choices)

    parser.add_argument('--dataset', type=str, default=dataset_choices[0], choices=dataset_choices, help='Dataset to choose from.')
    parser.add_argument('--vsize', type=int, default=100, help='Dimension of intrinsic parmaeter space.')
    parser.add_argument('--d_rate', '--dr', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--depth', type=int, default=2, help='Number of layers in FNN.')
    parser.add_argument('--width', type=int, default=100, help='Width of  layers in FNN.')
    parser.add_argument('--minibatch', '--mb', type=int, default=128, help='Size of minibatch.')
    parser.add_argument('--lr_ratio', '--lrr', type=float, default=.1, help='Ratio to decay LR by every LR_EPSTEP epochs.')
    parser.add_argument('--lr_epochs', '--lrep', type=float, default=0, help='Decay LR every LR_EPSTEP epochs. 0 to turn off decay.')
    parser.add_argument('--lr_steps', '--lrst', type=float, default=3, help='Max LR steps.')

    parser.add_argument('--c1', type=int, default=6, help='Channels in first conv layer, for LeNet.')
    parser.add_argument('--c2', type=int, default=16, help='Channels in second conv layer, for LeNet.')
    parser.add_argument('--d1', type=int, default=120, help='Channels in first dense layer, for LeNet.')
    parser.add_argument('--d2', type=int, default=84, help='Channels in second dense layer, for LeNet.')
    
    parser.add_argument('--denseproj', action='store_true', help='Use a dense projection.')
    parser.add_argument('--sparseproj', action='store_true', help='Use a sparse projection.')
    parser.add_argument('--fastfoodproj', action='store_true', help='Use a fastfood projection.')

    parser.add_argument('--partial_data', '--pd', type=float, default=1.0, help='Percentage of dataset.')

    parser.add_argument('--skiptfevents', action='store_true', help='Skip writing tf events files even if output is used.')

    args = parser.parse_args()

    n_proj_specified = sum([args.denseproj, args.sparseproj, args.fastfoodproj])
    if args.arch in arch_choices_projected:
        assert n_proj_specified == 1, 'Arch "%s" requires projection. Specify exactly one of {denseproj, sparseproj, fastfoodproj} options.' % args.arch
    else:
        assert n_proj_specified == 0, 'Arch "%s" does not require projection, so do not specify any of {denseproj, sparseproj, fastfoodproj} options.' % args.arch

    if args.denseproj:
        proj_type = 'dense'
    elif args.sparseproj:
        proj_type = 'sparse'
    else:
        proj_type = 'fastfood'

    train_style, val_style = ('', '') if args.nocolor else (colorama.Fore.BLUE, colorama.Fore.MAGENTA)

    # Get a TF session registered with Keras and set numpy and TF seeds
    sess = setup_session_and_seeds(args.seed)

    # 0. LOAD DATA

    if args.dataset=='agnews':
        loadpath = "./dataset/ag_news.p"
        x = cPickle.load(open(loadpath, "rb"))
        train, val, test = x[0], x[1], x[2]
        train_lab, val_lab, test_lab = x[3], x[4], x[5]
        wordtoix, ixtoword = x[6], x[7]
        MAX_SEQUENCE_LENGTH = 108
        n_label = 4
        n_max_pool = 52
        n_stride = 1

        loadpath = "./dataset/ag_news_glove.p"
        x = cPickle.load(open(loadpath, "rb"))
        embedding_matrix = x[0]

        n_train, n_test = len(train_lab), len(test_lab)
        train_y_,  test_y_= np.zeros(n_train), np.zeros(n_test)
        for i in range(n_train):
            train_y_[i] = np.argmax(train_lab[i][:,0])
        for i in range(n_test):
            test_y_[i] = np.argmax(test_lab[i][:,0])

        train_x = train
        train_y = train_y_
        val_x = test
        val_y = test_y_

    elif args.dataset=='yelp':
        loadpath = "./dataset/yelp.p"
        x = cPickle.load(open(loadpath, "rb"))
        train, val, test = x[0], x[1], x[2]
        train_lab, val_lab, test_lab = x[3], x[4], x[5]
        wordtoix, ixtoword = x[6], x[7]
        del x

        loadpath = "./dataset/yelp_glove.p"
        x = cPickle.load(open(loadpath, "rb"))
        embedding_matrix = x[0]

        MAX_SEQUENCE_LENGTH = 305
        n_label = 2
        n_max_pool = 151
        n_stride = 2

        train_x = train
        train_y = train_lab
        val_x = test
        val_y = test_lab
    else:
        raise Exception('Unknown dataset: %s' % args.dataset)  
    #pdb.set_trace()

    EMBED_DIM= 300
    train_x = pad_sequences(train_x, maxlen=MAX_SEQUENCE_LENGTH)
    val_x = pad_sequences(val_x, maxlen=MAX_SEQUENCE_LENGTH)
    n_words = len(ixtoword)
    del x
    print('Total words: %d' % n_words)


    if args.partial_data < 1.0:
        n_train_ = int(train_y.size*args.partial_data)       
        n_test_  = int(val_y.size*args.partial_data)       
        train_x = train_x[:n_train_]
        train_y = train_y[:n_train_]
        val_x = val_x[:n_test_]
        val_y = val_y[:n_test_]

    # load into memory if less than 1 GB
    if len(train_x) * 4 + len(val_x) * 4 < 1e9:
        train_x, train_y = np.array(train_x), np.array(train_y)
        val_x, val_y = np.array(val_x), np.array(val_y)

    # 1. CREATE MODEL
    randmirrors = False
    randcrops = False
    cropsize = None

    with WithTimer('Make model'):
        if args.arch == 'mnistfc_dir':
            model = build_model_mnist_fc_dir(weight_decay=args.l2, depth=args.depth, width=args.width)
        elif args.arch == 'mnistfc':
            if proj_type == 'fastfood':
                model = build_model_mnist_fc_fastfood(weight_decay=args.l2, vsize=args.vsize, depth=args.depth, width=args.width)
            else:
                model = build_model_mnist_fc(weight_decay=args.l2, vsize=args.vsize, depth=args.depth, width=args.width, proj_type=proj_type)
        elif args.arch == 'swem_dir':  
            model = build_model_text_swem_dir(embedding_matrix=embedding_matrix, weight_decay=args.l2, n_words=n_words, depth=args.depth, width=args.width, seq_length=MAX_SEQUENCE_LENGTH, embed_dim=EMBED_DIM, n_label=n_label)
        elif args.arch == 'swem':
            if proj_type == 'fastfood':
                model = build_model_text_swem_fastfood(embedding_matrix=embedding_matrix, weight_decay=args.l2, vsize=args.vsize, n_words=n_words, depth=args.depth, width=args.width, seq_length=MAX_SEQUENCE_LENGTH, embed_dim=EMBED_DIM, n_label=n_label)
            else:
                model = build_model_text_swem(embedding_matrix=embedding_matrix, weight_decay=args.l2, vsize=args.vsize, n_words=n_words, depth=args.depth, width=args.width, proj_type=proj_type, seq_length=MAX_SEQUENCE_LENGTH, embed_dim=EMBED_DIM, n_label=n_label)
        elif args.arch == 'cnn_dir':
            model = build_model_text_cnn_dir(embedding_matrix=embedding_matrix, weight_decay=args.l2, n_words=n_words, depth=args.depth, width=args.width, seq_length=MAX_SEQUENCE_LENGTH, embed_dim=EMBED_DIM, n_label=n_label, n_max_pool=n_max_pool, n_stride=n_stride)   
        elif args.arch == 'cnn':
            if proj_type == 'fastfood':
                model = build_model_text_cnn_fastfood(embedding_matrix=embedding_matrix, weight_decay=args.l2, vsize=args.vsize, n_words=n_words, depth=args.depth, width=args.width, seq_length=MAX_SEQUENCE_LENGTH, embed_dim=EMBED_DIM, n_label=n_label, n_max_pool=n_max_pool, n_stride=n_stride)
            else:
                model = build_model_text_cnn(embedding_matrix=embedding_matrix, weight_decay=args.l2, vsize=args.vsize, n_words=n_words, depth=args.depth, width=args.width, proj_type=proj_type, seq_length=MAX_SEQUENCE_LENGTH, embed_dim=EMBED_DIM, n_label=n_label, n_max_pool=n_max_pool, n_stride=n_stride)     
        else:
            raise Exception('Unknown network architecture: %s' % args.arch)

    print 'All model weights:'
    total_params = summarize_weights(model.trainable_weights)
    print 'Model summary:'
    model.summary()

    model.print_trainable_warnings()

    input_lr = tf.placeholder(tf.float32, shape=[])
    lr_stepper = LRStepper(args.lr, args.lr_ratio, args.lr_epochs, args.lr_steps)
    
    # 2. COMPUTE GRADS AND CREATE OPTIMIZER
    if args.opt == 'sgd':
        opt = tf.train.MomentumOptimizer(input_lr, args.mom)
    elif args.opt == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(input_lr, momentum=args.mom)
    elif args.opt == 'adam':
        opt = tf.train.AdamOptimizer(input_lr, args.beta1, args.beta2)

    # Optimize w.r.t all trainable params in the model
    grads_and_vars = opt.compute_gradients(model.v.loss, model.trainable_weights, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    train_step = opt.apply_gradients(grads_and_vars)
    add_grad_summaries(grads_and_vars)
    summarize_opt(opt)

    # 3. OPTIONALLY SAVE OR LOAD VARIABLES (e.g. model params, model running BN means, optimization momentum, ...) and then finalize initialization
    saver = tf.train.Saver(max_to_keep=None) if (args.output or args.load) else None
    if args.load:
        ckptfile, miscfile = args.load.split(':')
        # Restore values directly to graph
        saver.restore(sess, ckptfile)
        with gzip.open(miscfile) as ff:
            saved = pickle.load(ff)
            buddy = saved['buddy']
    else:
        buddy = StatsBuddy()
    buddy.tic()    # call if new run OR resumed run

    # Initialize any missed vars (e.g. optimization momentum, ... if not loaded from checkpoint)
    uninitialized_vars = tf_get_uninitialized_variables(sess)
    init_missed_vars = tf.variables_initializer(uninitialized_vars, 'init_missed_vars')

    sess.run(init_missed_vars)

    # Print warnings about any TF vs. Keras shape mismatches
    warn_misaligned_shapes(model)
    # Make sure all variables, which are model variables, have been initialized (e.g. model params and model running BN means)
    tf_assert_all_init(sess)

    # 3.5 Normalize the overall basis matrix across the (multiple) unnormalized basis matrices for each layer
    basis_matrices = []
    normalizers = []
    
    for layer in model.layers:
        try:
            basis_matrices.extend(layer.offset_creator.basis_matrices)
        except AttributeError:
            continue
        try:
            normalizers.extend(layer.offset_creator.basis_matrix_normalizers)
        except AttributeError:
            continue

    if len(basis_matrices) > 0 and not args.load:

        if proj_type == 'sparse':

            # Norm of overall basis matrix rows (num elements in each sum == total parameters in model)
            bm_row_norms = tf.sqrt(tf.add_n([tf.sparse_reduce_sum(tf.square(bm), 1) for bm in basis_matrices]))
            # Assign `normalizer` Variable to these row norms to achieve normalization of the basis matrix
            # in the TF computational graph
            rescale_basis_matrices = [tf.assign(var, tf.reshape(bm_row_norms,var.shape)) for var in normalizers]
            _ = sess.run(rescale_basis_matrices)
        elif proj_type == 'dense':
            bm_sums = [tf.reduce_sum(tf.square(bm), 1) for bm in basis_matrices]
            divisor = tf.expand_dims(tf.sqrt(tf.add_n(bm_sums)), 1)
            rescale_basis_matrices = [tf.assign(var, var / divisor) for var in basis_matrices]
            _ = sess.run(rescale_basis_matrices)
        else:
            print '\nhere\n'
            embed()

            assert False, 'what to do with fastfood?'

    # 4. SETUP TENSORBOARD LOGGING
    train_histogram_summaries = get_collection_intersection_summary('train_collection', 'orig_histogram')
    train_scalar_summaries    = get_collection_intersection_summary('train_collection', 'orig_scalar')
    val_histogram_summaries   = get_collection_intersection_summary('val_collection', 'orig_histogram')
    val_scalar_summaries      = get_collection_intersection_summary('val_collection', 'orig_scalar')
    param_histogram_summaries = get_collection_intersection_summary('param_collection', 'orig_histogram')

    writer = None
    if args.output:
        mkdir_p(args.output)
        if not args.skiptfevents:
            writer = tf.summary.FileWriter(args.output, sess.graph)



    # 5. TRAIN
    train_iters = (len(train_y) - 1) / args.minibatch + 1
    val_iters = (len(val_y) - 1) / args.minibatch + 1
    impreproc = ImagePreproc()

    if args.ipy:
        print 'Embed: before train / val loop (Ctrl-D to continue)'
        embed()

    fastest_avg_iter_time = 1e9
    
    while buddy.epoch < args.epochs + 1:
        # How often to log data
        do_log_params = lambda ep, it, ii: False
        do_log_val = lambda ep, it, ii: True
        do_log_train = lambda ep, it, ii: (it < train_iters and it & it-1 == 0 or it>=train_iters and it%train_iters == 0)  # Log on powers of two then every epoch

        # 0. Log params
        if args.output and do_log_params(buddy.epoch, buddy.train_iter, 0) and param_histogram_summaries is not None and not args.skiptfevents:
            params_summary_str, = sess.run([param_histogram_summaries])
            writer.add_summary(params_summary_str, buddy.train_iter)

        # 1. Evaluate val set performance
        if not args.skipval:
            tic2()
            for ii in xrange(val_iters):
                start_idx = ii * args.minibatch
                batch_x = val_x[start_idx:start_idx + args.minibatch]
                batch_y = val_y[start_idx:start_idx + args.minibatch]

                if randcrops:
                    batch_x = impreproc.center_crops(batch_x, cropsize)
                feed_dict = {
                    model.v.input_sequences: batch_x,
                    # model.v.input_masks: batch_mask,
                    model.v.input_labels: batch_y,
                    K.learning_phase(): 0,
                }
                fetch_dict = model.trackable_dict
                with WithTimer('sess.run val iter', quiet=not args.verbose):
                    result_val = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)
                
                buddy.note_weighted_list(batch_x.shape[0], model.trackable_names, [result_val[k] for k in model.trackable_names], prefix='val_')

            if args.output and not args.skiptfevents and do_log_val(buddy.epoch, buddy.train_iter, 0):
                log_scalars(writer, buddy.train_iter,
                            {'mean_%s' % name: value for name, value in buddy.epoch_mean_list_re('^val_')},
                            prefix='buddy')

            print ('\ntime: %f. after training for %d epochs:\n%3d val:   %s (%.3gs/i)'
                   % (buddy.toc(), buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^val_', style=val_style), toc2() / val_iters))

        # 2. Possiby Snapshot, possibly quit
        if args.output and args.snapshot_to and args.snapshot_every:
            snap_intermed = args.snapshot_every > 0 and buddy.train_iter % args.snapshot_every == 0
            snap_end = buddy.epoch == args.epochs
            if snap_intermed or snap_end:
                # Snapshot
                save_path = saver.save(sess, '%s/%s_%04d.ckpt' % (args.output, args.snapshot_to, buddy.epoch))
                print 'snappshotted model to', save_path
                with gzip.open('%s/%s_misc_%04d.pkl.gz' % (args.output, args.snapshot_to, buddy.epoch), 'w') as ff:
                    saved = {'buddy': buddy}
                    pickle.dump(saved, ff)

        if buddy.epoch == args.epochs:
            if args.ipy:
                print 'Embed: at end of training (Ctrl-D to exit)'
                embed()
            break   # Extra pass at end: just report val stats and skip training

        # 3. Train on training set
        #train_order = range(train_x.shape[0])
        if args.shuffletrain:
            train_order = np.random.permutation(train_x.shape[0])

        tic3()
        for ii in xrange(train_iters):
            tic2()
            start_idx = ii * args.minibatch

            if args.shuffletrain:
                batch_x = train_x[train_order[start_idx:start_idx + args.minibatch]]
                batch_y = train_y[train_order[start_idx:start_idx + args.minibatch]]
            else:
                batch_x = train_x[start_idx:start_idx + args.minibatch]
                batch_y = train_y[start_idx:start_idx + args.minibatch]
            if randcrops:
                batch_x = impreproc.random_crops(batch_x, cropsize, randmirrors)


            feed_dict = {
                model.v.input_sequences: batch_x,
                # model.v.input_masks: batch_mask,
                model.v.input_labels: batch_y,
                input_lr: lr_stepper.lr(buddy),
                K.learning_phase(): 1,
            }

            fetch_dict = {'train_step': train_step}
            fetch_dict.update(model.trackable_and_update_dict)

            if args.output and not args.skiptfevents and do_log_train(buddy.epoch, buddy.train_iter, ii):
                if param_histogram_summaries is not None:
                    fetch_dict.update({'param_histogram_summaries': param_histogram_summaries})
                if train_histogram_summaries is not None:
                    fetch_dict.update({'train_histogram_summaries': train_histogram_summaries})
                if train_scalar_summaries is not None:
                    fetch_dict.update({'train_scalar_summaries': train_scalar_summaries})

            with WithTimer('sess.run train iter', quiet=not args.verbose):
                result_train = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)

            buddy.note_weighted_list(batch_x.shape[0], model.trackable_names, [result_train[k] for k in model.trackable_names], prefix='train_')

            if do_log_train(buddy.epoch, buddy.train_iter, ii):
                print ('%3d train: %s (%.3gs/i)' % (buddy.train_iter, buddy.epoch_mean_pretty_re('^train_', style=train_style), toc2()))
                if args.output and not args.skiptfevents:
                    if param_histogram_summaries is not None:
                        hist_summary_str = result_train['param_histogram_summaries']
                        writer.add_summary(hist_summary_str, buddy.train_iter)
                    if train_histogram_summaries is not None:
                        hist_summary_str = result_train['train_histogram_summaries']
                        writer.add_summary(hist_summary_str, buddy.train_iter)
                    if train_scalar_summaries is not None:
                        scalar_summary_str = result_train['train_scalar_summaries']
                        writer.add_summary(scalar_summary_str, buddy.train_iter)
                    log_scalars(writer, buddy.train_iter,
                                {'batch_%s' % name: value for name, value in buddy.last_list_re('^train_')},
                                prefix='buddy')

            if ii > 0 and ii % 100 == 0:
                avg_iter_time = toc3() / 100; tic3()
                fastest_avg_iter_time = min(fastest_avg_iter_time, avg_iter_time)
                print '  %d: Average iteration time over last 100 train iters: %.3gs' % (ii, avg_iter_time)

            buddy.inc_train_iter()   # after finished training a mini-batch

        buddy.inc_epoch()   # after finished training whole pass through set

        if args.output and not args.skiptfevents and do_log_train(buddy.epoch, buddy.train_iter, 0):
            log_scalars(writer, buddy.train_iter,
                        {'mean_%s' % name: value for name,value in buddy.epoch_mean_list_re('^train_')},
                        prefix='buddy')

    print '\nFinal'
    print '%02d:%d val:   %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^val_', style=val_style))
    print '%02d:%d train: %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^train_', style=train_style))

    print '\nfinal_stats epochs %g' % buddy.epoch
    print 'final_stats iters %g' % buddy.train_iter
    print 'final_stats time %g' % buddy.toc()
    print 'final_stats total_params %g' % total_params
    print 'final_stats fastest_avg_iter_time %g' % fastest_avg_iter_time
    for name, value in buddy.epoch_mean_list_all():
        print 'final_stats %s %g' % (name, value)

    if args.output and not args.skiptfevents:
        writer.close()   # Flush and close


if __name__ == '__main__':
    main()
