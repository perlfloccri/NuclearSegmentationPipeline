# !/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (C) Software Competence Center Hagenberg GmbH (SCCH)
# All rights reserved.
# -----------------------------------------------------------------------------
# This document contains proprietary information belonging to SCCH.
# Passing on and copying of this document, use and communication of its
# contents is not permitted without prior written authorization.
# -----------------------------------------------------------------------------
# Created on : 29.11.2016 10:36 $
# by : Fischer $
# SVN : $
#

# --- imports -----------------------------------------------------------------
from unet1 import *

import cv2
import numpy as np
from skimage.measure import label
from skimage import transform as tf
from lasagne_wrapper.batch_iterators import BatchIterator,h5BatchIterator
from lasagne_wrapper.training_strategy import RefinementStrategy

import sys
sys.path.append(r'E:\github_publication\Config')
from Config.Config import UNETSettings

use_weights = True
INPUT_SHAPE = [1, 256, 256]

def get_segmentation_crop_flip_batch_iterator(flip_left_right=True, flip_up_down=True, crop_size=None, rotate=None,
                                              use_weights=False):

    def prepare(x, y):
        #new
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        # new end
        if rotate is not None:
            x_rot = np.zeros_like(x)
            y_rot = np.zeros_like(y)

            rot_angle = np.random.randint(-rotate, rotate)
            for i in xrange(x.shape[0]):
                x_tmp = x[i]
                y_tmp = y[i]
                xr = tf.rotate(np.transpose(x[i], (1, 2, 0)), rot_angle)
                xr = np.transpose(xr, (2, 0, 1))
                yr = tf.rotate(y[i].squeeze(), rot_angle, order=0)
                yr = yr.reshape(-1, y.shape[2], y.shape[3])
                x_rot[i] = xr
                y_rot[i] = yr

            x = x_rot
            y = y_rot

        # flipping
        if flip_left_right:
            fl = np.random.randint(0, 2, x.shape[0])
            for i in xrange(x.shape[0]):
                if fl[i] == 1:
                    x[i] = x[i, :, :, ::-1]
                    y[i] = y[i, :, :, ::-1]

        if flip_up_down:
            fl = np.random.randint(0, 2, x.shape[0])
            for i in xrange(x.shape[0]):
                if fl[i] == 1:
                    x[i] = x[i, :, ::-1, :]
                    y[i] = y[i, :, ::-1, :]

        if crop_size is not None:
            x_crop = np.zeros((x.shape[0], x.shape[1], crop_size[0], crop_size[1]), dtype=np.float32)
            y_crop = np.zeros((y.shape[0], y.shape[1], crop_size[0], crop_size[1]), dtype=np.float32)
            for i in xrange(x.shape[0]):
                max_row = x.shape[2] - crop_size[0]
                max_col = x.shape[3] - crop_size[1]

                row_0 = np.random.randint(0, max_row)
                row_1 = row_0 + crop_size[0]
                col_0 = np.random.randint(0, max_col)
                col_1 = col_0 + crop_size[1]

                x_crop[i, :] = x[i, :, row_0:row_1, col_0:col_1]
                y_crop[i, :] = y[i, :, row_0:row_1, col_0:col_1]

            x = x_crop
            y = y_crop

        if use_weights:
            # prepare weights
            w = np.zeros_like(y, dtype=np.float32)
            MAX_WEIGHT = 5
            for i in xrange(x.shape[0]):
                I = y[i, 0].astype(np.uint8)

                L = label(I)
                All_D = np.zeros_like(L, dtype=np.float32)
                for l in np.unique(L)[1::]:
                    C = (L == l).astype(np.uint8)
                    D = cv2.distanceTransform(1 - C, distanceType=cv2.cv.CV_DIST_L2, maskSize=cv2.cv.CV_DIST_MASK_PRECISE)
                    D = D.max() - D
                    D /= D.max()
                    D = np.exp((5 * D) ** 2)

                    All_D += D

                # only divide if there are cells in the image (ALL_D.max() > 0)
                if All_D.max() > 0:
                    All_D /= All_D.max()
                All_D *= 8
                All_D += 2
                All_D[L > 0] = 1
                W = All_D

                w[i, 0] = W.astype(np.float32)

            return x, y, w
        else:
            return x, y

    def batch_iterator(batch_size, k_samples, shuffle):
        #return h5BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


# prepare training strategy
train_strategy = get_binary_segmentation_TrainingStrategy(batch_size=2,
                                                          max_epochs=int(UNETSettings().network_info["max_epochs"]),
                                                          #samples_per_epoch=250,
                                                          patience=50,#20
                                                          ini_learning_rate=0.001,
                                                          #L2=None,
                                                          use_weights=use_weights,
                                                          #refinement_strategy=RefinementStrategy(n_refinement_steps=6),
                                                          valid_batch_iter=get_batch_iterator(),
                                                          train_batch_iter=get_segmentation_crop_flip_batch_iterator(flip_left_right=True,
                                                                                                                     flip_up_down=True,
                                                                                                                     rotate=None,#30,##None, #45
                                                                                                                     use_weights=use_weights,
                                                                                                                     crop_size=None)) #INPUT_SHAPE[1:]))
                                                                                                                   #  ))