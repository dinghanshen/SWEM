import numpy as np
from IPython import embed
import pdb

import tensorflow as tf
from keras.layers import (Dense, Flatten, Input, Activation, Conv1D, MaxPooling1D,
                          Reshape, Dropout, Convolution2D, 
                          MaxPooling2D, BatchNormalization, 
                          Conv2D, GlobalAveragePooling2D, 
                          Concatenate, AveragePooling2D, 
                          LocallyConnected2D, Embedding, Lambda)
import keras.backend as K

from general.tfutil import hist_summaries_traintest, scalar_summaries_traintest

from keras_ext.engine import ExtendedModel
from keras_ext.layers import (RProjDense, 
                              RProjConv2D, 
                              RProjBatchNormalization, 
                              RProjLocallyConnected2D)
from keras_ext.rproj_layers_util import (OffsetCreatorDenseProj, 
                                         OffsetCreatorSparseProj, 
                                         OffsetCreatorFastfoodProj, 
                                         FastWalshHadamardProjector, 
                                         ThetaPrime)
from keras_ext.util import make_image_input_preproc
from densenet import transition, denseblock, transition_RProj, denseblock_RProj
from keras.regularizers import l2


def make_and_add_losses(model, input_labels):
    '''Add classification and L2 losses'''

    with tf.name_scope('losses') as scope:
        prob = tf.nn.softmax(model.v.logits, name='prob')
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.v.logits, labels=input_labels, name='cross_ent')
        loss_cross_ent = tf.reduce_mean(cross_ent, name='loss_cross_ent')
        model.add_trackable('loss_cross_ent', loss_cross_ent)
        class_prediction = tf.argmax(prob, 1)

        prediction_correct = tf.equal(class_prediction, input_labels, name='prediction_correct')
        accuracy = tf.reduce_mean(tf.to_float(prediction_correct), name='accuracy')
        model.add_trackable('accuracy', accuracy)
        hist_summaries_traintest(prob, cross_ent)
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


def build_model_text_swem_dir(embedding_matrix, weight_decay=0,n_words=1000,depth=2, width=100, seq_length=100, embed_dim=300, n_label=2):
    #im_shape = (28, 28, 1)
    MAX_SEQUENCE_LENGTH = seq_length # 108
    EMBEDDING_DIM = embed_dim
    n_label = n_label
    im_dtype = 'float32'

    # embedding_matrix = # TODO: load pre-train embeddings

    with tf.name_scope('inputs'):
        # input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        # input_labels = Input(batch_shape=(None,), dtype='int64')
        input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')

        input_labels = Input(batch_shape=(None,), dtype='int64')
        

    with tf.name_scope('net') as scope:
        embedding_layer = Embedding(n_words, #400002,
                                    EMBEDDING_DIM,
                                    # embeddings_initializer='uniform',
                                    weights=[embedding_matrix], 
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        def average_emb(input_seq):

            H_enc = tf.reduce_max(input_seq, axis=1)  # batch 1 emb

            embedded_sequences = H_enc
            return embedded_sequences

        xx = embedding_layer(input_sequences)
        
        for _ in range(depth):
            xx = Dense(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        xx = Lambda(average_emb)(xx)
        logits = Dense(n_label, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        #pdb.set_trace()

        model = ExtendedModel(input=input_sequences, output=logits)

    nontrackable_fields = ['input_sequences','input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])
    
    make_and_add_losses(model, input_labels)
    
    return model


def build_model_text_swem(embedding_matrix, weight_decay=0,n_words=1000, vsize=100, depth=2, width=100, proj_type='sparse', seq_length=100, embed_dim=300, n_label=2):
    #im_shape = (28, 28, 1)
    MAX_SEQUENCE_LENGTH = seq_length # 108
    EMBEDDING_DIM = embed_dim
    n_label = n_label
    im_dtype = 'float32'

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj

    with tf.name_scope('inputs'):
        # input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        # input_labels = Input(batch_shape=(None,), dtype='int64')
        input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')

        input_labels = Input(batch_shape=(None,), dtype='int64')


    with tf.name_scope('net') as scope:
        vv = ThetaPrime(vsize)


        embedding_layer = Embedding(n_words,
                                    EMBEDDING_DIM,
                                    # embeddings_initializer='uniform',
                                    weights=[embedding_matrix], 
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        def average_emb(input_seq):

            H_enc = tf.reduce_max(input_seq, axis=1)  # batch 1 emb

            embedded_sequences = H_enc
            return embedded_sequences

        xx = embedding_layer(input_sequences)
        xx = Lambda(average_emb)(xx)

        for _ in range(depth):
            xx = RProjDense(offset_creator_class, vv, width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = RProjDense(offset_creator_class, vv, n_label, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        model = ExtendedModel(input=input_sequences, output=logits)

        model.add_extra_trainable_weight(vv.var)

    nontrackable_fields = ['input_sequences','input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])
    
    make_and_add_losses(model, input_labels)
    
    return model



def build_model_text_swem_fastfood(embedding_matrix, weight_decay=0,n_words=1000, vsize=100, depth=2, width=100, seq_length=100, embed_dim=300, n_label=2, DD=None):
    '''If DD is not specified, it will be computed.'''

    MAX_SEQUENCE_LENGTH = seq_length # 108
    EMBEDDING_DIM = embed_dim
    n_label = n_label
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
        input_labels = Input(batch_shape=(None,), dtype='int64')


    def define_model(input_sequences, DenseLayer, Conv2DLayer):
        vv = ThetaPrime(vsize)
        embedding_layer = Embedding(n_words,
                                    EMBEDDING_DIM,
                                    # embeddings_initializer='uniform',
                                    weights=[embedding_matrix], 
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        def average_emb(input_seq):
            H_enc = tf.reduce_max(input_seq, axis=1)  # batch 1 emb
            embedded_sequences = H_enc
            return embedded_sequences

        xx = embedding_layer(input_sequences)
        xx = Lambda(average_emb)(xx)
        for _ in range(depth):
            xx = DenseLayer(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(n_label, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        model = ExtendedModel(input=input_sequences, output=logits)
        nontrackable_fields = ['input_sequences', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_sequences, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_sequences, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_sequences', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model



def build_model_text_cnn_dir(embedding_matrix, weight_decay=0, n_words=1000, depth=2, width=100, seq_length=100, embed_dim=300, n_label=2, n_max_pool=52, n_stride=1):
    # im_shape = (28, 28, 1)
    MAX_SEQUENCE_LENGTH = seq_length # 108
    EMBEDDING_DIM = embed_dim
    n_label = n_label
    im_dtype = 'float32'

    # embedding_matrix = # TODO: load pre-train embeddings

    with tf.name_scope('inputs'):
        # input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        # input_labels = Input(batch_shape=(None,), dtype='int64')
        input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')

        input_labels = Input(batch_shape=(None,), dtype='int64')

    with tf.name_scope('net') as scope:
        embedding_layer = Embedding(n_words,
                                    EMBEDDING_DIM,
                                    # embeddings_initializer='uniform',
                                    weights=[embedding_matrix], 
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        def pool_emb(input_seq):
            embedded_sequences = tf.expand_dims(input_seq,-1)
            return embedded_sequences


        xx = embedding_layer(input_sequences)
        xx = Lambda(pool_emb)(xx)

        xx = Convolution2D(128, kernel_size=(5,EMBEDDING_DIM), strides=n_stride, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((n_max_pool, 1))(xx)# 52
        # xx = Convolution2D(32, kernel_size=(5,1), strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        # xx = MaxPooling2D((3, 1))(xx)
        # xx = Convolution2D(32, kernel_size=(5,1), strides=1, init='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        # xx = MaxPooling2D((3, 1))(xx)

        # l_cov1 = Conv1D(128, 5, activation='relu')(xx)
        # l_pool1 = MaxPooling1D(3)(l_cov1)
        # l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
        # l_pool2 = MaxPooling1D(3)(l_cov2)
        # xx = Conv1D(128, 5, activation='relu')(l_pool2)
        # xx = MaxPooling1D(6)(xx)  # global max pooling
        xx = Flatten()(xx)


        for _ in range(depth):
            xx = Dense(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
                xx)
        logits = Dense(n_label, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        # pdb.set_trace()

        model = ExtendedModel(input=input_sequences, output=logits)

    nontrackable_fields = ['input_sequences', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model


def build_model_text_cnn(embedding_matrix, weight_decay=0, n_words=1000,  vsize=100, depth=2, width=100, proj_type='sparse', seq_length=100, embed_dim=300, n_label=2, n_max_pool=52, n_stride=1):
    # im_shape = (28, 28, 1)
    MAX_SEQUENCE_LENGTH = seq_length # 108
    EMBEDDING_DIM = embed_dim
    n_label = n_label

    im_dtype = 'float32'

    # embedding_matrix = # TODO: load pre-train embeddings

    assert proj_type in ('dense', 'sparse')
    if proj_type == 'dense':
        offset_creator_class = OffsetCreatorDenseProj
    else:
        # sparse
        offset_creator_class = OffsetCreatorSparseProj


    with tf.name_scope('inputs'):
        # input_images, preproc_images = make_image_input_preproc(im_shape, im_dtype, shift_in=shift_in)
        # input_labels = Input(batch_shape=(None,), dtype='int64')
        input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')

        input_labels = Input(batch_shape=(None,), dtype='int64')


    with tf.name_scope('net') as scope:

        vv = ThetaPrime(vsize)

        embedding_layer = Embedding(n_words,
                                    EMBEDDING_DIM,
                                    # embeddings_initializer='uniform',
                                    weights=[embedding_matrix], 
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        def pool_emb(input_seq):
            embedded_sequences = tf.expand_dims(input_seq,-1)
            return embedded_sequences

        xx = embedding_layer(input_sequences)
        xx = Lambda(pool_emb)(xx)

        xx = RProjConv2D(offset_creator_class, vv, 128, kernel_size=(5,EMBEDDING_DIM), strides=n_stride, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((n_max_pool, 1))(xx)
        # xx = RProjConv2D(offset_creator_class, vv, 32, kernel_size=(5,1), strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        # xx = MaxPooling2D((3, 1))(xx)
        # xx = RProjConv2D(offset_creator_class, vv, 32, kernel_size=(5,1), strides=1, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        # xx = MaxPooling2D((3, 1))(xx)

        xx = Flatten()(xx)


        for _ in range(depth):
            xx = RProjDense(offset_creator_class, vv, width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
                xx)
        logits = RProjDense(offset_creator_class, vv, n_label, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        # pdb.set_trace()

        model = ExtendedModel(input=input_sequences, output=logits)

        model.add_extra_trainable_weight(vv.var)





    nontrackable_fields = ['input_sequences', 'input_labels', 'logits']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model





def build_model_text_cnn_fastfood(embedding_matrix, weight_decay=0,n_words=1000, vsize=100, depth=2, width=100, seq_length=100, embed_dim=300, n_label=2, n_max_pool=52, n_stride=1, DD=None):
    '''If DD is not specified, it will be computed.'''

    MAX_SEQUENCE_LENGTH = seq_length # 108
    EMBEDDING_DIM = embed_dim
    n_label = n_label
    im_dtype = 'float32'

    with tf.name_scope('inputs'):
        input_sequences = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
        input_labels = Input(batch_shape=(None,), dtype='int64')


    def define_model(input_sequences, DenseLayer, Conv2DLayer):
        vv = ThetaPrime(vsize)
        embedding_layer = Embedding(n_words,
                                    EMBEDDING_DIM,
                                    # embeddings_initializer='uniform',
                                    weights=[embedding_matrix], 
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        def pool_emb(input_seq):
            embedded_sequences = tf.expand_dims(input_seq,-1)
            return embedded_sequences

        xx = embedding_layer(input_sequences)
        xx = Lambda(pool_emb)(xx)
        xx = Conv2DLayer(128, kernel_size=(5,EMBEDDING_DIM), strides=n_stride, kernel_initializer='he_normal', padding='valid', activation='relu', kernel_regularizer=l2(weight_decay))(xx)
        xx = MaxPooling2D((n_max_pool, 1))(xx)
        xx = Flatten()(xx)

        for _ in range(depth):
            xx = DenseLayer(width, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)
        logits = DenseLayer(n_label, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(xx)

        model = ExtendedModel(input=input_sequences, output=logits)
        nontrackable_fields = ['input_sequences', 'input_labels', 'logits']
        for field in ['logits']:
            model.add_var(field, locals()[field])
        return model

    if not DD:
        with tf.name_scope('net_disposable'):
            # Make disposable direct model
            model_disposable = define_model(input_sequences, Dense, Conv2D)

            DD = np.sum([np.prod(var.get_shape().as_list()) for var in model_disposable.trainable_weights])
            del model_disposable

    print '\nProjecting from d = %d to D = %d parameters\n' % (vsize, DD)

    with tf.name_scope('net'):
        # Make real RProj FWH model
        fwh_projector = FastWalshHadamardProjector(vsize, DD)

        DenseLayer = lambda *args, **kwargs: RProjDense(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)
        Conv2DLayer = lambda *args, **kwargs: RProjConv2D(OffsetCreatorFastfoodProj, fwh_projector, *args, **kwargs)

        model = define_model(input_sequences, DenseLayer, Conv2DLayer)
        fwh_projector.check_usage()

        for ww in fwh_projector.trainable_weights:
            model.add_extra_trainable_weight(ww)
        for ww in fwh_projector.non_trainable_weights:
            model.add_extra_non_trainable_weight(ww)

    nontrackable_fields = ['input_sequences', 'input_labels']
    for field in nontrackable_fields:
        model.add_var(field, locals()[field])

    make_and_add_losses(model, input_labels)

    return model






