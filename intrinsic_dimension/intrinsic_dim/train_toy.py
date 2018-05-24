#! /usr/bin/env python

import sys
import os
import time
import gzip
import cPickle as pickle
import numpy as np
import h5py
import pdb
from IPython import embed
import colorama
import tensorflow as tf
import keras.backend as K

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(1, lab_root)

from general.util import tic, toc, tic2, toc2, tic3, toc3, mkdir_p, WithTimer
from general.image_preproc import ImagePreproc
from data.mnist.loader import centered_normed_mnist
from general.tfutil import hist_summaries_train, get_collection_intersection, get_collection_intersection_summary, log_scalars, sess_run_dict, summarize_weights, summarize_opt, tf_assert_all_init, tf_get_uninitialized_variables, add_grad_summaries
from general.stats_buddy import StatsBuddy
from keras_ext.util import setup_session_and_seeds, warn_misaligned_shapes

from standard_parser import make_standard_parser

from model_builders import build_toy
from train import LRStepper



def main():
    parser = make_standard_parser('Low Rank Basis experiments.', skip_train=True, skip_val=True, arch_choices=['one'])

    parser.add_argument('--DD', type=int, default=1000, help='Dimension of full parameter space.')
    parser.add_argument('--vsize', type=int, default=100, help='Dimension of intrinsic parameter space.')
    parser.add_argument('--lr_ratio', '--lrr', type=float, default=.5, help='Ratio to decay LR by every LR_EPSTEP epochs.')
    parser.add_argument('--lr_epochs', '--lrep', type=float, default=0, help='Decay LR every LR_EPSTEP epochs. 0 to turn off decay.')
    parser.add_argument('--lr_steps', '--lrst', type=float, default=3, help='Max LR steps.')
    
    parser.add_argument('--denseproj', action='store_true', help='Use a dense projection.')

    parser.add_argument('--skiptfevents', action='store_true', help='Skip writing tf events files even if output is used.')

    args = parser.parse_args()

    if args.denseproj:
        proj_type = 'dense'
    else:
        proj_type = None

    train_style, val_style = ('', '') if args.nocolor else (colorama.Fore.BLUE, colorama.Fore.MAGENTA)

    # Get a TF session registered with Keras and set numpy and TF seeds
    sess = setup_session_and_seeds(args.seed)

    # 1. CREATE MODEL

    with WithTimer('Make model'):
        if args.denseproj:
            model = build_toy(weight_decay=args.l2, DD=args.DD, groups=10, vsize=args.vsize, proj=True)
        else:
            model = build_toy(weight_decay=args.l2, DD=args.DD, proj=False)

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


    # Choose between sparsified and dense projection matrix if using them
    #SparseRM = True

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
            sess.run(rescale_basis_matrices)
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
    train_iters = 1
    val_iters = 1

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
                with WithTimer('val iter %d/%d'%(ii, val_iters), quiet=not args.verbose):
                    feed_dict = {
                        K.learning_phase(): 0,
                    }
                    fetch_dict = model.trackable_dict
                    with WithTimer('sess.run val iter', quiet=not args.verbose):
                        result_val = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)

                    buddy.note_weighted_list(1, model.trackable_names, [result_val[k] for k in model.trackable_names], prefix='val_')

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
        tic3()
        for ii in xrange(train_iters):

            with WithTimer('train iter %d/%d'%(ii, train_iters), quiet=not args.verbose):
            
                tic2()
                feed_dict = {
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

                buddy.note_weighted_list(1, model.trackable_names, [result_train[k] for k in model.trackable_names], prefix='train_')

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
                        log_scalars(writer, buddy.train_iter, {'batch_lr': lr_stepper.lr(buddy)}, prefix='buddy')

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
