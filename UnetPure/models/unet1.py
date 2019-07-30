#!/usr/bin/env python

import lasagne
from lasagne.layers.conv import Conv2DLayer as Conv2DLayer
from lasagne.layers import MaxPool2DLayer, ConcatLayer, TransposedConv2DLayer
from lasagne.nonlinearities import elu, sigmoid, rectify
from lasagne.layers import batch_norm

from lasagne_wrapper.network import SegmentationNetwork

from lasagne_wrapper.training_strategy import get_binary_segmentation_TrainingStrategy,get_categorical_segmentation_TrainingStrategy
from lasagne_wrapper.batch_iterators import get_batch_iterator
from lasagne_wrapper.learn_rate_shedules import get_stepwise
from lasagne_wrapper.parameter_updates import get_update_momentum

Network = SegmentationNetwork


INPUT_SHAPE = [1, 256, 256]
nonlin = elu


def conv_bn(in_layer, num_filters, filter_size, nonlinearity=rectify, pad='same', name='conv'):
    """ convolution block with with batch normalization """
    in_layer = Conv2DLayer(in_layer, num_filters=num_filters, filter_size=filter_size,
                           nonlinearity=nonlinearity, pad=pad, name=name)
    in_layer = batch_norm(in_layer)
    return in_layer


def build_model():
    """ Compile net architecture """

    l_in = lasagne.layers.InputLayer(shape=(None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), name='Input')
    net1 = batch_norm(l_in)

    # --- preprocessing ---
    net1 = conv_bn(net1, num_filters=10, filter_size=1, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=1, filter_size=1, nonlinearity=nonlin, pad='same', name='color_deconv_preproc')

    # number of filters in first layer
    # decreased by factor 2 in each block
    nf0 = 16

    # --- encoder ---
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    p1 = net1
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2, name='pool1')

    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    p2 = net1
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2, name='pool2')

    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    p3 = net1
    net1 = MaxPool2DLayer(net1, pool_size=2, stride=2, name='pool3')

    net1 = conv_bn(net1, num_filters=8 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=8 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')

    # --- decoder ---
    net1 = TransposedConv2DLayer(net1, num_filters=4 * nf0, filter_size=2, stride=2, name='upconv')
    net1 = ConcatLayer((p3, net1), name='concat')
    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=4 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')

    net1 = TransposedConv2DLayer(net1, num_filters=2 * nf0, filter_size=2, stride=2, name='upconv')
    net1 = ConcatLayer((p2, net1), name='concat')
    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=2 * nf0, filter_size=3, nonlinearity=nonlin, pad='same')

    net1 = TransposedConv2DLayer(net1, num_filters=nf0, filter_size=2, stride=2, name='upconv')
    net1 = ConcatLayer((p1, net1), name='concat')
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')
    net1 = conv_bn(net1, num_filters=nf0, filter_size=3, nonlinearity=nonlin, pad='same')

    net1 = Conv2DLayer(net1, num_filters=1, filter_size=1, nonlinearity=sigmoid, pad='same', name='segmentation')

    return net1


# prepare training strategy
train_strategy = get_binary_segmentation_TrainingStrategy(batch_size=2, max_epochs=1000, samples_per_epoch=250, patience=300,
                                                          ini_learning_rate=0.2, L2=None, use_weights=False,
                                                          adapt_learn_rate=get_stepwise(k=1000, factor=0.5),
                                                          update_function=get_update_momentum(0.9),
                                                          valid_batch_iter=get_batch_iterator(),
                                                          train_batch_iter=get_batch_iterator())
