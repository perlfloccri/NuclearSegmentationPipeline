
import numpy as np
from lasagne_wrapper.batch_iterators import BatchIterator


def get_batch_iterator():
    """
    Modified batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        def prepare(X, Y):
            X = X.astype(np.float32) / 255
            Y = Y.astype(np.float32)
            return X, Y

        return BatchIterator(batch_size=batch_size, k_samples=k_samples, shuffle=shuffle, prepare=prepare)

    return batch_iterator


def get_weighted_batch_iterator(WEIGHT_BALLS=1e1):
    """
    Modified batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        def prepare(X, Y):
            X = X.astype(np.float32) / 255
            Y = Y.astype(np.float32)

            W = np.ones_like(Y)
            W[Y == 1] = WEIGHT_BALLS
            for i in xrange(W.shape[0]):

                # random left right flips
                if np.random.randint(0, 2) > 0:
                    X[i] = X[i, :, :, ::-1]
                    Y[i] = Y[i, :, :, ::-1]
                    W[i] = W[i, :, :, ::-1]

                W[i] /= W[i].sum()
                W[i] *= np.prod(W.shape[2:])

            return X, Y, W

        return BatchIterator(batch_size=batch_size, k_samples=k_samples, shuffle=shuffle, prepare=prepare)

    return batch_iterator


def get_crop_batch_iterator():
    """
    Modified batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        def prepare(X, Y):
            X = X.astype(np.float32) / 255
            Y = Y.astype(np.float32)

            # get random crop of full image
            r, c = X.shape[2:]
            r_2 = r // 2
            c_2 = c // 2

            X_crop = np.zeros((batch_size, 1, r_2, c_2), dtype=np.float32)
            Y_crop = np.zeros((batch_size, 1, r_2, c_2), dtype=np.float32)

            for i in xrange(batch_size):
                r0 = np.random.randint(low=0, high=r_2)
                c0 = np.random.randint(low=0, high=c_2)
                r1 = r0 + r_2
                c1 = c0 + c_2
                X_crop[i, 0] = X[i, 0, r0:r1, c0:c1]
                Y_crop[i, 0] = Y[i, 0, r0:r1, c0:c1]

            W = np.ones_like(Y_crop)
            W[Y_crop == 1] = 1e1
            for i in xrange(W.shape[0]):
                W[i] /= W[i].sum()
                W[i] *= np.prod(W.shape[2:])

            return X_crop, Y_crop, W

        return BatchIterator(batch_size=batch_size, k_samples=k_samples, shuffle=shuffle, prepare=prepare)

    return batch_iterator


def get_loc_batch_iterator(N_BINS, flip=False):
    """
    Modified batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):

        def prepare(X, Y):
            from utils.data import prepare_targets
            X = X.astype(np.float32) / 255
            Y = Y.astype(np.float32)

            T = np.zeros((Y.shape[0], np.sum(N_BINS)), dtype=np.float32)
            for i in xrange(Y.shape[0]):

                # random left right flips
                if flip and np.random.randint(0, 2) > 0:
                    X[i] = X[i, :, :, ::-1]
                    Y[i] = Y[i, :, :, ::-1]

                T[i] = prepare_targets(Y[i, 0], Y[i, 0].shape, n_buckets=N_BINS)

            return X, T

        return BatchIterator(batch_size=batch_size, k_samples=k_samples, shuffle=shuffle, prepare=prepare)

    return batch_iterator