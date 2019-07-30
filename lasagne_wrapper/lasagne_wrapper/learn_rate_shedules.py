
import numpy as np
from lasagne.utils import floatX


def get_constant():
    """
    Keep learning rate constant
    """
    def update(lr, epoch):
        return lr

    return update


def get_stepwise(k=10, factor=0.5):
    """
    Stepwise learning rate update every k epochs
    """
    def update(lr, epoch):

        if epoch >= 0 and np.mod(epoch, k) == 0:
            return floatX(factor * lr)
        else:
            return floatX(lr)

    return update


def get_stepwise(k=10, factor=0.5):
    """
    Stepwise learning rate after no update for several epochs
    """
    def update(lr, epoch):

        if epoch >= 0 and np.mod(epoch, k) == 0:
            return floatX(factor * lr)
        else:
            return floatX(lr)

    return update
