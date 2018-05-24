#! /usr/bin/env python

"""
This training script explores multi-gpu distributed training using Horovod (https://github.com/uber/horovod)
Needs NCCL library and installation of Open MPI

Run like this

mpirun -np 4 ./train_distrbuted.py /data_local/imagenet/h5/train.h5 /data_local/imagenet/h5/val.h5 -E 100

"""

import sys
import os
import time
import gzip
import cPickle as pickle
import numpy as np
import h5py
from IPython import embed
import colorama
import tensorflow as tf
import keras.backend as K

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(1, lab_root)

from general.util import tic, toc, tic2, toc2, tic3, toc3, mkdir_p, WithTimer
from general.image_preproc import ImagePreproc
from general.stats_buddy import StatsBuddy
from general.tfutil import (get_collection_intersection_summary, 
                            log_scalars, sess_run_dict, 
                            summarize_weights, summarize_opt, 
                            tf_get_uninitialized_variables)
from model_builders import (build_alexnet_direct, 
                            build_alexnet_fastfood, 
                            build_squeezenet_direct, 
                            build_squeezenet_fastfood)
from standard_parser import make_standard_parser
import horovod.tensorflow as hvd

arch_choices_direct = ['alexnet_dir', 'squeeze_dir']
arch_choices_projected = ['alexnet', 'squeeze']
arch_choices = arch_choices_direct + arch_choices_projected

def main():
    parser = make_standard_parser('Distributed Training of Direct or RProj model on Imagenet',
                                  arch_choices=arch_choices)

    parser.add_argument('--vsize', type=int, default=100, help='Dimension of intrinsic parmaeter space.')
    parser.add_argument('--minibatch', '--mb', type=int, default=256, help='Size of minibatch.')
    parser.add_argument('--denseproj', action='store_true', help='Use a dense projection.')
    parser.add_argument('--sparseproj', action='store_true', help='Use a sparse projection.')
    parser.add_argument('--fastfoodproj', action='store_true', help='Use a fastfood projection.')

    args = parser.parse_args()

    minibatch_size = args.minibatch
    train_style, val_style = ('', '') if args.nocolor else (colorama.Fore.BLUE, colorama.Fore.MAGENTA)

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
    
    # Initialize Horovod
    hvd.init()

    #minibatch_size = 256
    worker_minibatch_size = minibatch_size / hvd.size()
    
    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement=True
    my_rank = hvd.local_rank()
    print "I am worker ", my_rank
    
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    # Adjust number of epochs based on number of GPUs.
    epochs = args.epochs 

    # Add hook to broadcast variables from rank 0 to all other processes during
    # initialization.
    #hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.


    # 0. LOAD DATA
    train_h5 = h5py.File(args.train_h5, 'r')
    train_x = train_h5['images']
    train_y = train_h5['labels']
    val_h5 = h5py.File(args.val_h5, 'r')
    val_x = val_h5['images']
    val_y = val_h5['labels']

    # load into memory if less than 1 GB
    if train_x.size * 4 + val_x.size * 4 < 1e9:
        train_x, train_y = np.array(train_x), np.array(train_y)
        val_x, val_y = np.array(val_x), np.array(val_y)


    # 1. CREATE MODEL
    extra_feed_dict = {}

    with WithTimer('Make model'):
        if args.arch == 'alexnet_dir':
            shift_in = np.array([104, 117, 123], dtype='float32')
            model = build_alexnet_direct(weight_decay=args.l2, shift_in=shift_in)
            randmirrors = True 
            randcrops = True
            cropsize = (227,227)

        elif args.arch == 'squeeze_dir':
            model = build_squeezenet_direct(weight_decay=args.l2, shift_in=np.array([104, 117, 123]))
            randmirrors = True
            randcrops = True
            cropsize = (224,224)

        elif args.arch == 'alexnet':
            if proj_type == 'fastfood':
                model = build_alexnet_fastfood(weight_decay=args.l2, shift_in=np.array([104, 117, 123]), vsize=args.vsize)
            else:
                raise Exception('not implemented')
            randmirrors = True
            randcrops = True
            cropsize = (227,227)
        
        elif args.arch == 'squeeze':
            if proj_type == 'fastfood':
                model = build_squeezenet_fastfood(weight_decay=args.l2, shift_in=np.array([104, 117, 123]), vsize=args.vsize)
            else:
                raise Exception('not implemented')
            randmirrors = True
            randcrops = True
            cropsize = (224,224)

        else:
            raise Exception('Unknown network architecture: %s' % args.arch)


    if my_rank == 0:
        print 'All model weights:'
        summarize_weights(model.trainable_weights)
        print 'Model summary:'
        model.summary()
        model.print_trainable_warnings()


    lr = args.lr

    if args.opt == 'sgd':
        opt = tf.train.MomentumOptimizer(lr, args.mom)
    elif args.opt == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(lr, momentum=args.mom)
    elif args.opt == 'adam':
        opt = tf.train.AdamOptimizer(lr, args.beta1, args.beta2)

    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_step = opt.minimize(model.v.loss, global_step=global_step)
        
    sess = K.get_session()
    sess.run(hvd.broadcast_global_variables(0))

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

    # 4. SETUP TENSORBOARD LOGGING
    train_histogram_summaries = get_collection_intersection_summary('train_collection', 'orig_histogram')
    train_scalar_summaries    = get_collection_intersection_summary('train_collection', 'orig_scalar')
    val_histogram_summaries   = get_collection_intersection_summary('val_collection', 'orig_histogram')
    val_scalar_summaries      = get_collection_intersection_summary('val_collection', 'orig_scalar')
    param_histogram_summaries = get_collection_intersection_summary('param_collection', 'orig_histogram')

    writer = None
    if args.output:
        mkdir_p(args.output)
        writer = tf.summary.FileWriter(args.output, sess.graph)


    ## 5. TRAIN
    train_iters = (train_y.shape[0] - 1) / minibatch_size
    val_iters = (val_y.shape[0] - 1) / minibatch_size
    impreproc = ImagePreproc()

    if args.ipy:
        print 'Embed: before train / val loop (Ctrl-D to continue)'
        embed()

    while buddy.epoch < args.epochs + 1:
        # How often to log data
        do_log_params = lambda ep, it, ii: True
        do_log_val = lambda ep, it, ii: True
        do_log_train = lambda ep, it, ii: (it < train_iters and it & it-1 == 0 or it>=train_iters and it%train_iters == 0)  # Log on powers of two then every epoch

        # 0. Log params
        if args.output and do_log_params(buddy.epoch, buddy.train_iter, 0) and param_histogram_summaries is not None:
            params_summary_str, = sess.run([param_histogram_summaries])
            writer.add_summary(params_summary_str, buddy.train_iter)

        # 1. Evaluate val set performance
        if not args.skipval:
            tic2()
            for ii in xrange(val_iters):
                with WithTimer('(worker %d) val iter %d/%d'%(my_rank, ii, val_iters), quiet=not args.verbose):

                    start_idx = ii * minibatch_size
                    
                    # each worker gets a portion of the minibatch
                    my_start = start_idx + my_rank * worker_minibatch_size
                    my_end = my_start + worker_minibatch_size

                    batch_x = val_x[my_start:my_end]
                    batch_y = val_y[my_start:my_end]

                    #print "**** I am worker %d, my val batch starts %d and ends %d"%(my_rank, my_start, my_end)

                    if randcrops:
                        batch_x = impreproc.center_crops(batch_x, cropsize)
                    feed_dict = {
                        model.v.input_images: batch_x,
                        model.v.input_labels: batch_y,
                        K.learning_phase(): 0,
                    }
                    feed_dict.update(extra_feed_dict)
                    fetch_dict = model.trackable_dict
                    with WithTimer('(worker %d) sess.run val iter'%my_rank, quiet=not args.verbose):
                        result_val = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)

                    buddy.note_weighted_list(batch_x.shape[0], model.trackable_names, [result_val[k] for k in model.trackable_names], prefix='val_')

            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0):
                log_scalars(writer, buddy.train_iter,
                            {'mean_%s' % name: value for name, value in buddy.epoch_mean_list_re('^val_')},
                            prefix='buddy')

            print ('\ntime: %f. after training for %d epochs:\n%3d (worker %d) val:   %s (%.3gs/i)'
                   % (buddy.toc(), buddy.epoch, buddy.train_iter, my_rank, buddy.epoch_mean_pretty_re('^val_', style=val_style), toc2() / val_iters))

        # 2. Possiby Snapshot, possibly quit
        # only worker 0 handles it
        if args.output and args.snapshot_to and args.snapshot_every:
            snap_intermed = args.snapshot_every > 0 and buddy.train_iter % args.snapshot_every == 0
            snap_end = buddy.epoch == args.epochs
            if snap_intermed or snap_end:
                # Snapshot
                if my_rank == 0:
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
        tic3()
        for ii in xrange(train_iters):
            tic2()

            with WithTimer('(worker %d) train iter %d/%d'%(my_rank, ii, train_iters), quiet=not args.verbose):

                if args.shuffletrain:
                    start_idx = np.random.randint(train_x.shape[0]-minibatch_size)
                else:
                    start_idx = ii * minibatch_size
                
                # each worker gets a portion of the minibatch
                my_start = start_idx + my_rank * worker_minibatch_size
                my_end = my_start + worker_minibatch_size
                
                #print "**** ii is %d, train_iters is %d"%(ii, train_iters)
                #print "**** I am worker %d, my training batch starts %d and ends %d (total: %d)"%(my_rank, my_start, my_end, train_x.shape[0])

                batch_x = train_x[my_start:my_end]
                batch_y = train_y[my_start:my_end]
                    
                if randcrops:
                    batch_x = impreproc.random_crops(batch_x, cropsize, randmirrors)

                feed_dict = {
                    model.v.input_images: batch_x,
                    model.v.input_labels: batch_y,
                    K.learning_phase(): 1,
                }
                feed_dict.update(extra_feed_dict)

                fetch_dict = {'train_step': train_step}
                fetch_dict.update(model.trackable_and_update_dict)

                if args.output and do_log_train(buddy.epoch, buddy.train_iter, ii):
                    if param_histogram_summaries is not None:
                        fetch_dict.update({'param_histogram_summaries': param_histogram_summaries})
                    if train_histogram_summaries is not None:
                        fetch_dict.update({'train_histogram_summaries': train_histogram_summaries})
                    if train_scalar_summaries is not None:
                        fetch_dict.update({'train_scalar_summaries': train_scalar_summaries})

                with WithTimer('(worker %d) sess.run train iter'%my_rank, quiet=not args.verbose):
                    result_train = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)

                buddy.note_weighted_list(batch_x.shape[0], model.trackable_names, [result_train[k] for k in model.trackable_names], prefix='train_')

                if do_log_train(buddy.epoch, buddy.train_iter, ii):
                    print ('%3d (worker %d) train: %s (%.3gs/i)' % (buddy.train_iter, my_rank, buddy.epoch_mean_pretty_re('^train_', style=train_style), toc2()))
                    if args.output:
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

                if ii > 0 and ii % 100 == 0: print '  %d: Average iteration time over last 100 train iters: %.3gs' % (ii, toc3() / 100); tic3()

                buddy.inc_train_iter()   # after finished training a mini-batch

        buddy.inc_epoch()   # after finished training whole pass through set

        if args.output and do_log_train(buddy.epoch, buddy.train_iter, 0):
            log_scalars(writer, buddy.train_iter,
                        {'mean_%s' % name: value for name,value in buddy.epoch_mean_list_re('^train_')},
                        prefix='buddy')

    print '\nFinal'
    print '%02d:%d val:   %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^val_', style=val_style))
    print '%02d:%d train: %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^train_', style=train_style))

    print '\nfinal_stats epochs %g' % buddy.epoch
    print 'final_stats iters %g' % buddy.train_iter
    print 'final_stats time %g' % buddy.toc()
    for name, value in buddy.epoch_mean_list_all():
        print 'final_stats %s %g' % (name, value)

    if args.output:
        writer.close()   # Flush and close


if __name__ == '__main__':
    main()
