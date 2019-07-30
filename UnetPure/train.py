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
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
sys.path.append(r'E:\github_publication\Config')
from Datahandling.Datasets import TisquantDatasetNew,ArtificialNucleiDataset,ArtificialNucleiDatasetNotConverted,MergedDataset, TisquantDataset
from Config.Config import UNETSettings
from TileImages.tools import write_to_list_h5py
from utils.cell_data import prepare_data
import h5py
from models import unet1_augment as model
base_path = r"E:\github_publication\UnetPure"

if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--model', help='select model to train.', default="models\unet1_augment.py")
    parser.add_argument('--datadescription', help='select data set.',default="256x256_TisQuantTrainingData_Evaluation1")
    args = parser.parse_args()



    # Load settings
    settings = UNETSettings()
    # Load Dataset
    print("Load dataset ...")
    if UNETSettings().network_info["dataset"] == 'tisquant':
        dataset = TisquantDatasetNew()
        # dataset = TisquantDataset()
    elif UNETSettings().network_info["dataset"] == 'artificialNuclei':
        dataset = ArtificialNucleiDataset()
    elif UNETSettings().network_info["dataset"] == 'artificialNucleiNotConverted':
        dataset = ArtificialNucleiDatasetNotConverted()
    elif UNETSettings().network_info["dataset"] == 'mergeTisquantArtificialNotConverted':
        datasets = []
        dataset1 = TisquantDatasetNew()
        dataset1.load_data(mode=1)
        dataset2 = ArtificialNucleiDatasetNotConverted()
        dataset2.load_data(mode=1)
        datasets.append(dataset1)
        datasets.append(dataset2)
        dataset = MergedDataset(datasets)
    elif UNETSettings().network_info["dataset"] == 'mergeTisquantArtificial':
        datasets = []
        dataset1 = TisquantDatasetNew()
        dataset1.load_data(mode=1)
        dataset2 = ArtificialNucleiDataset()
        dataset2.load_data(mode=1)
        datasets.append(dataset1)
        datasets.append(dataset2)
        dataset = MergedDataset(datasets)
    else:
        print('Dataset not valid')
        sys.exit("Error")

    # Load Dataset
    dataset.load_data(mode=1)
    dataset.prepare()

    # get the model
    net = model.build_model()

    # load train data
    print("loading data")

    data = prepare_data(dataset)

    """
    #new: write to h5py file
    write_to_list_h5py(os.path.join("tmp", "Unetclassic_trainingdata.h5"), data)
    del data
    # Reload data
    h5_data = h5py.File(os.path.join("tmp", "Unetclassic_trainingdata.h5"), 'r')
    #new end
    """
    # initialize neural network
    my_net = model.SegmentationNetwork(net)

    # train network

    model_id = settings.network_info["net_description"]
    model_name = "UNet_Classic_" + model_id;
    dump_file = os.path.join(base_path, "model_params", model_name + ".pkl")
    log_file = os.path.join(base_path, "model_params", model_name + "_log.pkl")

    my_net.fit(data, model.train_strategy, dump_file=dump_file, log_file=log_file)
    #my_net.fit(h5_data['data'], model.train_strategy, dump_file=dump_file, log_file=log_file)
