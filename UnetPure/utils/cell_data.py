
import os
import cv2
import glob
import numpy as np
import pickle
import skimage.measure as skiMeas
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.io import loadmat, savemat
from PIL import Image
from PIL import ImageEnhance

import h5py
import scipy
import sys
sys.path.append(r'E:\github_publication\Config')
from TileImages.tools import write_to_list_h5py
from tqdm import tqdm
import skimage.transform as ski_transform
import re
input_shape = [1, 256, 256]

root_dir = r"E:\github_publication\UnetPure\data"
result_dir = os.path.join(root_dir, 'dl_seg_results')
DATASET_FOLDER = r'D:\DeepLearning\SCCHCode\TisQuantValidation\data'
DATASETROOT = 'D:\DeepLearning\IASS\Mask_RCNN\CVSP\Set1'

def pre_process_img(img, color):
    """
    Preprocess image
    """
    if color is 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color is 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        pass
    
    img = img.astype(np.float32)
    img /= 255.0
    
    return img

def prepare_data(dataset):

    Images, Masks = [], []
    n_tr = dataset.train_cnt;

    print("Preparing data for combined prediction ...")
    for i in tqdm(dataset.image_ids):
        img = dataset.load_image(i)
        mask = dataset.load_mask_one_layer(i).astype(np.uint8)
        new_mask = np.zeros_like(mask).astype(np.uint8)

        #tmp_mask = scipy.ndimage.label(mask > 0)[0]
        tmp_mask = mask
        labels = np.unique(tmp_mask)
        for label in labels[1::]:
            cell_mask = (tmp_mask == label).astype(np.uint16)
            cell_mask = cv2.dilate(cell_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            new_mask[cell_mask != 0] += 1
        fixed_mask = tmp_mask != 0
        fixed_mask[new_mask > 1] = 0
        mask = fixed_mask.astype(np.uint8)
        Images.append(img)
        Masks.append(mask>0)

    img_size = Images[0].shape
    Images = np.asarray(Images, dtype=np.uint8).reshape(-1, 1, img_size[0], img_size[1])
    write_to_list_h5py(os.path.join("tmp", "train_img.h5"), Images)
    del Images
    Masks = np.asarray(Masks, dtype=np.uint8).reshape(-1, 1, img_size[0], img_size[1])
    write_to_list_h5py(os.path.join("tmp", "train_msk.h5"), Masks)
    del Masks

    # load h5py
    h5_img = h5py.File(os.path.join("tmp", "train_img.h5"), 'r')
    h5_msk = h5py.File(os.path.join("tmp", "train_msk.h5"), 'r')

    data_to_ret = dict()
    data_to_ret['X_train'] = np.asarray(h5_img['data'][0:n_tr], dtype=np.uint8)
    data_to_ret['X_test'] = np.asarray(h5_img['data'][n_tr::],dtype=np.uint8)
    data_to_ret['X_valid'] = np.asarray(h5_img['data'][n_tr::], dtype=np.uint8)
    data_to_ret['y_train'] = h5_msk['data'][0:n_tr]
    data_to_ret['y_test'] = h5_msk['data'][n_tr::]
    data_to_ret['y_valid'] = h5_msk['data'][n_tr::]
    return data_to_ret

    """
    Images, Masks = [], []
    n_tr = dataset.train_cnt;

    print("Preparing data for combined prediction ...")
    for i in tqdm(dataset.image_ids):
        img = dataset.load_image(i)
        mask = dataset.load_mask_one_layer(i).astype(np.float32)
        new_mask = np.zeros_like(mask).astype(np.int)
        tmp_mask = scipy.ndimage.label(mask > 0)[0]
        labels = np.unique(tmp_mask)
        for label in labels[1::]:
            cell_mask = (tmp_mask == label).astype(np.uint16)
            cell_mask = cv2.dilate(cell_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            new_mask[cell_mask != 0] += 1
        fixed_mask = tmp_mask != 0
        fixed_mask[new_mask > 1] = 0
        mask = fixed_mask.astype(np.float16)
        Images.append(img)
        Masks.append(mask>0)
        
    # convert to conv net format
    img_size = Images[0].shape
    Images = np.asarray(Images, dtype=np.float32).reshape(-1, 1, img_size[0], img_size[1])
    write_to_list_h5py(os.path.join("tmp", "train_img.h5"), Images)
    del Images
    Masks = np.asarray(Masks, dtype=np.float32).reshape(-1, 1, img_size[0], img_size[1])
    write_to_list_h5py(os.path.join("tmp", "train_msk.h5"), Masks)
    del Masks

    # load h5py
    h5_img = h5py.File(os.path.join("tmp", "train_img.h5"), 'r')
    h5_msk = h5py.File(os.path.join("tmp", "train_msk.h5"), 'r')

    data_to_ret = dict()
    data_to_ret['X_train'] = np.asarray(h5_img['data'][0:n_tr], dtype=np.float32)
    data_to_ret['X_test'] = np.asarray(h5_img['data'][n_tr::],dtype=np.float32)
    data_to_ret['X_valid'] = np.asarray(h5_img['data'][n_tr::], dtype=np.float32)
    data_to_ret['y_train'] = h5_msk['data'][0:n_tr]
    data_to_ret['y_test'] = h5_msk['data'][n_tr::]
    data_to_ret['y_valid'] = h5_msk['data'][n_tr::]
    return data_to_ret
    """

def prepare_data_for_evaluation(dataset): # mode 1 ... load training data for training       mode 2 ... load training data to predict L1 for next level

    Images = []
    for i in tqdm(dataset.image_ids):
        Images.append(dataset.load_image(i))

    img_size = Images[0].shape
    Images = np.asarray(Images, dtype=np.float32).reshape(-1, 1, img_size[0], img_size[1])  # ,img_size[2])
    data = dict(X_train=Images)

    return data