
from __future__ import print_function

import numpy as np


def get_batch_iterator():
    """
    Standard batch iterator
    """
    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def get_flip_batch_iterator(flip_left_right=True, flip_up_down=False):
    """
    Image classification batch iterator which randomly flips images left and right
    """
    
    def prepare(x, y):

        # flipping
        if flip_left_right:
            fl = np.random.randint(0, 2, x.shape[0])
            try:
                xr = xrange(x.shape[0])
            except NameError:
                xr = range(x.shape[0])
            for i in xr:
                if fl[i] == 1:
                    x[i] = x[i, :, :, ::-1]

        if flip_up_down:
            fl = np.random.randint(0, 2, x.shape[0])
            try:
                xr = xrange(x.shape[0])
            except NameError:
                xr = range(x.shape[0])
            for i in xr:
                if fl[i] == 1:
                    x[i] = x[i, :, ::-1, :]

        return x, y
    
    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def get_segmentation_flip_batch_iterator(flip_left_right=True, flip_up_down=False):
    """
    Image segmentation batch iterator which randomly flips images (and mask) left and right
    """

    def prepare(x, y):

        # flipping
        if flip_left_right:
            fl = np.random.randint(0, 2, x.shape[0])
            try:
                xr = xrange(x.shape[0])
            except NameError:
                xr = range(x.shape[0])
            for i in xr:
                if fl[i] == 1:
                    x[i] = x[i, :, :, ::-1]
                    y[i] = y[i, :, :, ::-1]

        if flip_up_down:
            fl = np.random.randint(0, 2, x.shape[0])
            try:
                xr = xrange(x.shape[0])
            except NameError:
                xr = range(x.shape[0])
            for i in xr:
                if fl[i] == 1:
                    x[i] = x[i, :, ::-1, :]
                    y[i] = y[i, :, ::-1, :]

        return x, y

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


class BatchIterator(object):
    """
    Prototype for batch iterator
    """

    def __init__(self, batch_size, re_iterate=1, prepare=None, k_samples=None, shuffle=False):
        self.batch_size = batch_size

        if prepare is None:
            def prepare(*data):
                return data
        self.prepare = prepare
        
        self.re_iterate = re_iterate
        self.k_samples = k_samples
        self.shuffle = shuffle

    def __call__(self, data_pool):
        self.data_pool = data_pool
        if self.k_samples is None:
            self.k_samples = self.data_pool.shape[0]
        self.n_batches = self.re_iterate * (self.k_samples // self.batch_size)
        return self

    def __iter__(self):

        if self.shuffle:
            self.data_pool.shuffle()
        
        # reiterate entire data-set
        try:
            xr = xrange(self.re_iterate)
        except NameError:
            xr = range(self.re_iterate)
        for _ in xr:
                
            # use only k samples per epoch
            try:
                xr = xrange((self.k_samples + self.batch_size - 1) / self.batch_size)
            except NameError:
                xr = range(int((self.k_samples + self.batch_size - 1) / self.batch_size))
            for i_b in xr:

                # slice batch data
                start = i_b * self.batch_size
                stop = (i_b + 1) * self.batch_size
                sl = slice(start, stop)
                xb = self.data_pool[sl]

                # get missing samples
                n_sampels = xb[0].shape[0]
                if n_sampels < self.batch_size:
                    n_missing = self.batch_size - n_sampels

                    x_con = self.data_pool[0:n_missing]
                    try:
                        xr = xrange(len(xb))
                    except NameError:
                        xr = range(len(xb))
                    for i_input in xr:
                        xb[i_input] = np.concatenate((xb[i_input], x_con[i_input]))

                yield self.transform(xb)
    
    def transform(self, data):
        return self.prepare(*data)


class h5BatchIterator(BatchIterator):

    def __call__(self, data_pool):
        self.data_pool = data_pool
        self.data_ind = np.arange(data_pool.shape[0])
        if self.k_samples is None:
            self.k_samples = self.data_pool.shape[0]
        self.n_batches = self.re_iterate * (self.k_samples // self.batch_size)
        return self

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data_ind)

        # reiterate entire data-set
        try:
            xr = xrange(self.re_iterate)
        except NameError:
            xr = range(self.re_iterate)
        for _ in xr:

            # use only k samples per epoch
            try:
                xr = xrange((self.k_samples + self.batch_size - 1) / self.batch_size)
            except NameError:
                xr = range(int((self.k_samples + self.batch_size - 1) / self.batch_size))
            for i_b in xr:

                # slice batch data
                start = i_b * self.batch_size
                stop = (i_b + 1) * self.batch_size
                sl = slice(start, stop)
                xb = self.data_pool[self.data_ind[sl]]

                # get missing samples
                n_sampels = xb[0].shape[0]
                if n_sampels < self.batch_size:
                    n_missing = self.batch_size - n_sampels

                    x_con = self.data_pool[self.data_ind[0:n_missing]]
                    try:
                        xr = xrange(len(xb))
                    except NameError:
                        xr = range(len(xb))
                    for i_input in xr:
                        xb[i_input] = np.concatenate((xb[i_input], x_con[i_input]))

                yield self.transform(xb)

    def transform(self, data):
        return self.prepare(*data)

class stratifiedBatchIterator(BatchIterator):
    def transform(self, data):
        return self.prepare(*data)

    def __iter__(self):
        batchsize = self.batch_size
        shuffle = self.shuffle
        forever=False
        inputs = self.data_pool[:][0]
        cellsizes = self.data_pool[:][1]
        bins = self.data_pool[:][2]
        classes=np.unique(bins)
        nclass=len(classes)
        ninputs=[]
        nindices=[]
        nbins=[]
        nlen=[]
        nix=[]
        for cls in classes:
            ix = np.where(bins == cls)[0]
            nix.append(ix)
            if shuffle:
                np.random.shuffle(ix)
            ninputs.append(inputs[ix])
            nindices.append(np.arange(len(inputs)))
            nbins.append(bins[ix])
            nlen.append(len(ix))

        while True:

            excerpts=[]
            nexcerpts = []
            for i, cls in enumerate(classes):
                # excerpt = np.empty((0,), dtype=np.int32)
                excerpt=[]
                for start_idx in range(0, nlen[i] - (batchsize / nclass) + 1, (batchsize / nclass)):
                    ix_ = slice(start_idx, start_idx + (batchsize / nclass))
                    ex = nix[i][ix_]
                    # excerpt = np.append(excerpts, ex, axis=0)
                    excerpt.append(ex)
                excerpts.append(excerpt)
                nexcerpts.append(len(excerpt))

            min_nexpt = min(nexcerpts)
            max_nexpt = max(nexcerpts)

            for ex in range(min_nexpt):
                final_excerpt = []
                for i, cls in enumerate(classes):
                    final_excerpt.append(excerpts[i][ex])
                final_excerpt_=np.concatenate(final_excerpt)
                yield self.transform((inputs[final_excerpt_], cellsizes[final_excerpt_]))
            if not forever:
                break

def threaded_generator(generator, num_cached=10):
    """
    Threaded generator
    """
    try:
        import Queue
    except ImportError:
        import queue as Queue

    queue = Queue.Queue(maxsize=num_cached)
    queue = Queue.Queue(maxsize=num_cached)
    end_marker = object()

    # define producer
    def producer():
        for item in generator:
            #item = np.array(item)  # if needed, create a copy here
            queue.put(item)
        queue.put(end_marker)

    # start producer
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def generator_from_iterator(iterator):
    """
    Compile generator from iterator
    """
    for x in iterator:
        yield x


def threaded_generator_from_iterator(iterator, num_cached=10):
    """
    Compile threaded generator from iterator
    """
    generator = generator_from_iterator(iterator)
    return threaded_generator(generator, num_cached)


def batch_compute1(X, compute, batch_size):
    """ Batch compute data """

    # init results
    R = None

    # get number of samples
    n_samples = X.shape[0]

    # get input shape
    in_shape = list(X.shape)[1:]

    # get number of batches
    n_batches = int(np.ceil(float(n_samples) / batch_size))

    # iterate batches
    try:
        xr = xrange(n_batches)
    except NameError:
        xr = range(n_batches)
    for i_batch in xr:

        # extract batch
        start_idx = i_batch * batch_size
        excerpt = slice(start_idx, start_idx + batch_size)
        E = X[excerpt]

        # append zeros if batch is to small
        n_missing = batch_size - E.shape[0]
        if n_missing > 0:
            E = np.vstack((E, np.zeros([n_missing] + in_shape, dtype=X.dtype)))

        # compute results on batch
        r = compute(E)

        # init result array
        if R is None:
            R = np.zeros([n_samples] + list(r.shape[1:]), dtype=r.dtype)

        # store results
        R[start_idx:start_idx+r.shape[0]] = r[0:batch_size-n_missing]

    return R


if __name__ == '__main__':
    """ main """
    from data_pool import DataPool

    # init some random data
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    # init data pool
    data_pool = DataPool(X, y)
    iterator = BatchIterator(batch_size=7)
    for x, y in iterator(data_pool):
        print(x.shape, y.shape)

    # init data pool
    data_pool = DataPool(X)
    iterator = BatchIterator(batch_size=7)
    for x, in iterator(data_pool):
        print(x.shape)
