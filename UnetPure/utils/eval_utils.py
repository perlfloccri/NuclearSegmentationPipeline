
import numpy as np


def dice(Seg, G):
    """ compute dice coefficient """

    if (np.sum(G) + np.sum(Seg)) == 0:
        dice = 1.0
    else:
        dice = (2.0 * np.sum(Seg[G == 1])) / (np.sum(Seg) + np.sum(G))

    return dice
