#!/usr/bin/env python
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
#from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as skiMeas
import pickle
import h5py
import skimage.transform as ski_transform
from tqdm import tqdm, trange
input_shape = [1, 256, 256]

from scipy.io import loadmat, savemat
from kaggle import get_kaggle_data
sys.path.append(r'E:\github_publication\Config')
from Datahandling.Datasets import TisquantDatasetNew,ArtificialNucleiDataset,ArtificialNucleiDatasetNotConverted,MergedDataset, TisquantDataset, SampleInference
from Config.Config import UNETSettings
from utils.cell_data import prepare_data_for_evaluation
base_path = r"E:\github_publication\UnetPure"
input_shape = [1, 256, 256]

def select_model(model_str):

    """ select model for training """
    if os.path.isfile(model_str):
        model_name = model_str.split(os.path.sep)[1].split(".py")[0]
        exec "from models import " + model_name + " as model"
    else:
        pass
    return model


if __name__ == '__main__':
    """ main """

    # set test image id here!
    test_img_id = 1
    thresh = 0.4

    # add argument parser

    parser = argparse.ArgumentParser(description='Trained model.')
    parser.add_argument('--model', help='select model to test.', default="models\unet1_augment.py")
    parser.add_argument('--datadescription', help='select data set.',default="256x256_TisQuantTrainingData_Evaluation1")
    parser.add_argument('--testdata', help='select data set.',default="256x256_TisQuantTestData_Evaluation1")
    parser.add_argument('--path_to_img', help='select data set.',default=r"G:\FORSCHUNG\LAB4\Daria Lazic\Deep_Learning\GD2_META_rwf")
    parser.add_argument('--rwf', default="1")
    parser.add_argument('--mode', default=3)
    args = parser.parse_args()

    settings = UNETSettings()

    # Load Dataset
    print ("Load dataset ...")
    if settings.network_info["dataset"] == 'tisquant':  # args.dataset
        dataset = TisquantDatasetNew()
    elif UNETSettings().network_info["dataset"] == 'sampleInference':
        dataset = SampleInference()
    else:
        print('Dataset not valid')
        sys.exit("Error")

    val_idx = dataset.load_data(mode=2)
    dataset.prepare()

    # get the model
    exec "from models import " + "unet1_augment" + " as model"
    net = model.build_model()

    # initialize neural network
    my_net = model.SegmentationNetwork(net, print_architecture=False)

    # load network parameters
    model_id = settings.network_info["net_description"]
    model_name = "UNet_Classic_" + model_id;
    dump_file = os.path.join(base_path, "model_params", model_name + ".pkl")
    my_net.load(dump_file)

    data = prepare_data_for_evaluation(dataset)
    P = []
    for x in data['X_train']:
        P.append(my_net.predict_proba(x.reshape(-1, x.shape[0], x.shape[1], x.shape[2])))

    f = h5py.File(os.path.join(settings.network_info["results_folder"],
                               settings.network_info["net_description"] + "_predictions.h5"), 'a')
    f.create_dataset('predictions', data=P, dtype=np.float32)
    f.close()
