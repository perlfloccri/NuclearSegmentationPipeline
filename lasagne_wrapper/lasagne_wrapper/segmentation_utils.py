# -*- coding: utf-8 -*-

import lasagne
import numpy as np


def dice(Seg, GT):
    """ compute dice coefficient between current segmentation result and groundtruth (GT)"""

    if (np.sum(GT) + np.sum(Seg)) == 0:
        dice = 1.0
    else:
        dice = (2.0 * np.sum(Seg[GT == 1])) / (np.sum(Seg) + np.sum(GT))

    return dice


def pixelwise_softmax(net):
    """
    Apply pixelwise softmax
    """
    
    # get number of classes
    Nc = net.output_shape[1]
    
    # reshape 2 softmax
    shape = net.output_shape
    net = lasagne.layers.ReshapeLayer(net, shape=(-1, Nc, shape[2]*shape[3]))
    net = lasagne.layers.DimshuffleLayer(net, (0,2,1))
    net = lasagne.layers.ReshapeLayer(net, shape=(-1, Nc))

    net = lasagne.layers.NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)    

    # reshape 2 image
    net = lasagne.layers.ReshapeLayer(net, shape=(-1, shape[2]*shape[3], Nc))
    net = lasagne.layers.DimshuffleLayer(net, (0,2,1))
    net = lasagne.layers.ReshapeLayer(net, shape=(-1,shape[1],shape[2],shape[3]))
    
    return net