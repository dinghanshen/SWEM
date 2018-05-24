#! /usr/bin/env python

import sys
import os
import time
import gzip
import cPickle as pickle
import numpy as np
from IPython import embed
import colorama
import tensorflow as tf
import keras.backend as K

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(1, lab_root)

from general.util import mkdir_p, WithTimer
from general.image_preproc import ImagePreproc
from general.tfutil import (summarize_weights, 
                            summarize_opt, 
                            tf_assert_all_init, 
                            tf_get_uninitialized_variables)
from general.stats_buddy import StatsBuddy
from keras_ext.util import setup_session_and_seeds, warn_misaligned_shapes
from model_builders_rl import build_model_fc_dir, build_model_fc, build_model_fourier
from standard_parser import make_standard_parser
import gym
import itertools
import math

from collections import deque
from rl.neural_q_learner import NeuralQLearner

arch_choices = ['fc_dir', 'fc', 'fourier']
env_choices = ['CartPole-v0', 'CartPole-v1', 'CartPole-v2', 'CartPole-v3', 'CartPole-v4', 'CartPole-v5', 'MountainCar-v0','OffSwitchCartpole-v0','Acrobot-v1','Pendulum-v0']

def main():
    parser = make_standard_parser('Random Projection RL experiments.', arch_choices=arch_choices, skip_train=True, skip_val=True)

    parser.add_argument('--vsize', type=int, default=100, help='Dimension of intrinsic parmaeter space.')
    parser.add_argument('--depth', type=int, default=2, help='Number of layers in FNN.')
    parser.add_argument('--width', type=int, default=100, help='Width of layers in FNN.')
    parser.add_argument('--env_name', type=str, default=arch_choices[0], 
        choices=env_choices, help='Which architecture to use (choices: %s).' % env_choices)
    args = parser.parse_args()

    proj_type='dense'

    if args.arch == 'fourier':
        d_Fourier = 200

    train_style, val_style = ('', '') if args.nocolor else (colorama.Fore.BLUE, colorama.Fore.MAGENTA)

    # Get a TF session registered with Keras and set numpy and TF seeds
    sess = setup_session_and_seeds(args.seed)

    # 0. LOAD ENV
    theta_r = 2.0
    theta_threshold_radians = 12 * 2 * math.pi / 360  
    x_threshold = 1

    env_name = args.env_name
    env = gym.make(env_name)

    #sess = tf.Session()
    state_dim   = env.observation_space.shape[0]
    if env_name == 'Pendulum-v0':
        num_actions = 1 
    else:
        num_actions = env.action_space.n

    # 1. CREATE MODEL
    extra_feed_dict = {}
    with WithTimer('Make model'):
        if args.arch == 'fc_dir':
            model = build_model_fc_dir(state_dim, num_actions, weight_decay=args.l2, depth=args.depth, width=args.width, shift_in=None)
        elif args.arch == 'fc':
            model = build_model_fc(state_dim, num_actions, weight_decay=args.l2,  vsize=args.vsize, depth=args.depth, width=args.width, shift_in=None, proj_type='dense')
        elif args.arch == 'fourier':
            model = build_model_fourier(d_Fourier, num_actions, weight_decay=args.l2, depth=args.depth, width=args.width, shift_in=None)        
        else:
            raise Exception('Unknown network architecture: %s' % args.arch)    

    print 'All model weights:'
    summarize_weights(model.trainable_weights)
    print 'Model summary:'
    model.summary()
    model.print_trainable_warnings()


    # 2. COMPUTE GRADS AND CREATE OPTIMIZER
    if args.opt == 'sgd':
        optimizer = tf.train.MomentumOptimizer(args.lr, args.mom)
    elif args.opt == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(args.lr, momentum=args.mom)
    elif args.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(args.lr, args.beta1, args.beta2)
    
    summarize_opt(optimizer)

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

    # 3.5 Fourier features of the observations if required
    if args.arch == 'fourier':
        RP = np.random.randn(d_Fourier, state_dim)
        Rb = np.random.uniform( -math.pi, math.pi, d_Fourier)
        state_dim = d_Fourier


    # 4. SETUP TENSORBOARD LOGGING

    writer = None
    if args.output:
        mkdir_p(args.output)
        writer = tf.summary.FileWriter(args.output+"/{}-experiment-1".format(env_name), sess.graph)
    
    observation_to_action = model

    # 5. TRAIN
    if env_name == 'CartPole-v0': 

        q_learner = NeuralQLearner(sess,
                                   optimizer,
                                   observation_to_action,
                                   state_dim,
                                   num_actions,
                                   init_exp=0.5, 
                                   anneal_steps=10000, # N steps for annealing exploration
                                   discount_factor=0.95,
                                   batch_size=32,
                                   target_update_rate=0.01,
                                   summary_writer=writer)

        MAX_EPISODES = 10000
        MAX_STEPS    = 200

        episode_history = deque(maxlen=100)

        if args.ipy:
            print 'Embed: before train / val loop (Ctrl-D to continue)'
            embed()

        episode_history = deque(maxlen=100)
        for i_episode in range(MAX_EPISODES):

            # initialize
            state = env.reset()
            if args.arch == 'fourier':
                state = np.sin(RP.dot(state) + Rb) # np.median( RP.dot(state) )


            
            total_rewards = 0

            for t in range(MAX_STEPS):
                # env.render()
                action = q_learner.eGreedyAction(state[np.newaxis,:])
                next_state, reward, done, _ = env.step(action)

                if args.arch == 'fourier':
                    next_state = np.sin(RP.dot(next_state) + Rb) # np.median( RP.dot(state) )

                total_rewards += reward
                # reward = -10 if done else 0.1 # normalize reward
                q_learner.storeExperience(state, action, reward, next_state, done)

                q_learner.updateModel()
                state = next_state

                if done: break

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)
                
            print("Episode {}".format(i_episode))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

            if mean_rewards >= 195.0:
                print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
                break


    elif env_name == 'CartPole-v1': 

        q_learner = NeuralQLearner(sess,
                                   optimizer,
                                   observation_to_action,
                                   state_dim,
                                   num_actions,
                                   init_exp=0.5, 
                                   anneal_steps=10000, # N steps for annealing exploration
                                   discount_factor=0.95,
                                   batch_size=32,
                                   target_update_rate=0.01,
                                   summary_writer=writer)

        MAX_EPISODES = 10000
        MAX_STEPS    = 500

        episode_history = deque(maxlen=100)

        if args.ipy:
            print 'Embed: before train / val loop (Ctrl-D to continue)'
            embed()

        episode_history = deque(maxlen=100)
        for i_episode in range(MAX_EPISODES):

            # initialize
            state = env.reset()
            total_rewards = 0

            for t in itertools.count():
                # env.render()
                action = q_learner.eGreedyAction(state[np.newaxis,:])
                next_state, reward, done, _ = env.step(action)

                total_rewards += reward
                # reward = -10 if done else 0.1 # normalize reward
                q_learner.storeExperience(state, action, reward, next_state, done)

                q_learner.updateModel()
                state = next_state

                if done: break

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)
                
            print("Episode {}".format(i_episode))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

            if mean_rewards >= 475.0:
                print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
                break

    elif env_name == 'MountainCar-v0': 

        q_learner = NeuralQLearner(sess,
                               optimizer,
                               observation_to_action,
                               state_dim,
                               num_actions,
                               batch_size=64,
                               anneal_steps=10000, # N steps for annealing exploration
                               replay_buffer_size=1000000,
                               discount_factor=0.95,  # discount future rewards
                               target_update_rate=0.01,
                               reg_param=0.01, # regularization constants
                               summary_writer=writer)       

        MAX_EPISODES = 10000

        episode_history = deque(maxlen=100)

        if args.ipy:
            print 'Embed: before train / val loop (Ctrl-D to continue)'
            embed()

        episode_history = deque(maxlen=100)
        for i_episode in range(MAX_EPISODES):

            # initialize
            state = env.reset()
            total_rewards = 0
            for t in itertools.count():
                # env.render()
                action = q_learner.eGreedyAction(state[np.newaxis,:])
                next_state, reward, done, _ = env.step(action)

                total_rewards += reward
                # reward = -10 if done else 0.1 # normalize reward
                q_learner.storeExperience(state, action, reward, next_state, done)

                q_learner.updateModel()
                state = next_state

                if done: break

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)
                
            print("Episode {}".format(i_episode))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

            if mean_rewards >= -110.0:
                print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
                break

    elif env_name == 'CartPole-v2' or env_name == 'CartPole-v3'  or env_name == 'CartPole-v4'  or env_name == 'CartPole-v5':

        q_learner = NeuralQLearner(sess,
                                   optimizer,
                                   observation_to_action,
                                   state_dim,
                                   num_actions,
                                   init_exp=0.5, 
                                   anneal_steps=10000, # N steps for annealing exploration
                                   discount_factor=0.95,
                                   batch_size=32,
                                   target_update_rate=0.01,
                                   summary_writer=writer) 

        MAX_EPISODES = 10000
        MAX_STEPS    = 200

        episode_history = deque(maxlen=100)

        if args.ipy:
            print 'Embed: before train / val loop (Ctrl-D to continue)'
            embed()

        episode_history = deque(maxlen=100)
        for i_episode in range(MAX_EPISODES):

            # initialize
            state = env.reset()
            
            if args.arch == 'fourier':
                state = np.sin(RP.dot(state) + Rb) # np.median( RP.dot(state) )


            
            total_rewards = 0

            for t in range(MAX_STEPS):
                # env.render()
                action = q_learner.eGreedyAction(state[np.newaxis,:])

                next_state, reward, done, _ = env.step(action)

                if args.arch == 'fourier':
                    next_state = np.sin(RP.dot(next_state) + Rb) # np.median( RP.dot(state) )

                total_rewards += reward
                # reward = -10 if done else 0.1 # normalize reward
                q_learner.storeExperience(state, action, reward, next_state, done)

                q_learner.updateModel()
                state = next_state

                if done: break

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)
                
            print("Episode {}".format(i_episode))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

            if mean_rewards >= 195.0:
                print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
                break

    elif env_name == 'Pendulum-v0': 

        q_learner = NeuralQLearner(sess,
                                   optimizer,
                                   observation_to_action,
                                   state_dim,
                                   num_actions,
                                   init_exp=0.5, 
                                   anneal_steps=10000, # N steps for annealing exploration
                                   discount_factor=0.99,
                                   batch_size=32,
                                   target_update_rate=0.01,
                                   summary_writer=writer)     

        MAX_EPISODES = 10000
        MAX_STEPS    = 200

        episode_history = deque(maxlen=100)

        if args.ipy:
            print 'Embed: before train / val loop (Ctrl-D to continue)'
            embed()

        episode_history = deque(maxlen=100)
        for i_episode in range(MAX_EPISODES):

            # initialize
            state = env.reset()
            if args.arch == 'fourier':
                state = np.sin(RP.dot(state) + Rb) # np.median( RP.dot(state) )


            
            total_rewards = 0

            for t in range(MAX_STEPS):
                # env.render()
                action = q_learner.eGreedyAction(state[np.newaxis,:])
                next_state, reward, done, _ = env.step(action)
                if args.arch == 'fourier':
                    next_state = np.sin(RP.dot(next_state) + Rb) # np.median( RP.dot(state) )

                total_rewards += reward
                # reward = -10 if done else 0.1 # normalize reward
                q_learner.storeExperience(state, action, reward, next_state, done)

                q_learner.updateModel()
                state = next_state

                if done: break

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)
                
            print("Episode {}".format(i_episode))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

            if mean_rewards >= 195.0:
                print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
                break



if __name__ == '__main__':
    main()
