
from __future__ import print_function

import lasagne
import numpy as np


# init color printer
class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        """ Constructor """
        pass

    #    def print_colored(self, string, color):
    #        """ Change color of string """
    #        return color + string + BColors.ENDC

    def print_colored(self, string, color):
        """ Change color of string """
        return string


def print_net_architecture(net):
    """ print netarchitecture to command line """
    
    print("Network Architecture:")
    
    # get list of layers
    layers = lasagne.layers.helper.get_all_layers(net)
    
    # maximum layer name length
    max_len = np.max([len(l.__class__.__name__) for l in layers])
    
    for l in layers[::-1]:
        print(l.__class__.__name__.ljust(max_len), ":", l.output_shape)
    
    # print number of model paramters
    n_params = lasagne.layers.helper.count_params(net)
    print("\nNumber of Model Parameters:", n_params)
