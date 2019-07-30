# --- imports -----------------------------------------------------------------
from __future__ import print_function
import os
import numpy as np
import re
import argparse
#from lasagne_wrapper.network import Network
#from lasagne_wrapper.data_pool import DataPool
#from cell_classif_model_augment import build_model_meta, train_strategy_meta, build_model, train_strategy, load_model
import skimage.transform as ski_transform
from tqdm import tqdm
import pandas as pd
from skimage import filters
import glob
import cv2
import h5py

import matplotlib.pyplot as plt
INPUT_SHAPE = [3,256,256]

def rescaleAndTile (images=None,masks=None,scales=None,modality='image',mode='training',forunetorconvnet='UNET', rescale=True,overlap=20):
    img_to_return = []
    scales_to_return = []
    original_shape_to_return = []
    if (mode=='training'):
        MEAN_NUCLEI_SIZE = np.mean(scales)
    else:
        MEAN_NUCLEI_SIZE = np.load(os.path.join("tmp","mean_nuclear_size.npy"))

    if (modality == 'image'):
        nr_images = images.__len__()
    else:
        nr_images = masks.__len__()
    if (mode=='eval'):
        image_nr = []
    for i in tqdm(range(nr_images)):
        if (rescale):
            if (modality=='image'):
                image  = np.float32(ski_transform.resize(images[i], (int(images[i].shape[0] * 1 / (scales[i] / MEAN_NUCLEI_SIZE)), int(images[i].shape[1] * 1 / (scales[i] / MEAN_NUCLEI_SIZE))), mode='reflect'))
            else:
                image = rescale_mask(masks[i],int(masks[i].shape[0] * 1 / (scales[i] / MEAN_NUCLEI_SIZE)), int(masks[i].shape[1] * 1 / (scales[i] / MEAN_NUCLEI_SIZE)))
        else:
            if (modality=='image'):
                image = images[i]
                if ((forunetorconvnet == 'CONVNET') and (mode=='training')):
                    mask = masks[i]
                    scale = scales[i]
            else:
                image = masks[i]
        x_running = 0
        img_new = []
        thresh_img = []
        slicesize = [INPUT_SHAPE[1],INPUT_SHAPE[2],INPUT_SHAPE[0]]

        if (modality=='image'):
            [y, x, z] = image.shape
            try:
                if (rescale):
                    thresh_img.append(np.mean(image[:, :, 0][np.where(image[:, :, 0] < filters.threshold_otsu(image[:, :, 0]))]))
                    thresh_img.append(np.mean(image[:, :, 1][np.where(image[:, :, 1] < filters.threshold_otsu(image[:, :, 1]))]))
                    thresh_img.append(np.mean(image[:, :, 2][np.where(image[:, :, 2] < filters.threshold_otsu(image[:, :, 2]))]))
                else:
                    thresh_img.append(int(np.mean(image[:, :, 0][np.where(image[:, :, 0] < filters.threshold_otsu(image[:, :, 0]))])))
                    thresh_img.append(int(np.mean(image[:, :, 1][np.where(image[:, :, 1] < filters.threshold_otsu(image[:, :, 1]))])))
                    thresh_img.append(int(np.mean(image[:, :, 2][np.where(image[:, :, 2] < filters.threshold_otsu(image[:, :, 2]))])))
            except:
                thresh_img = []
                thresh_img.append(0)
                thresh_img.append(0)
                thresh_img.append(0)
        else:
            [y, x] = image.shape
        while (x_running <= x):
            y_running = 0
            while (y_running <= y):
                min_x_orig = x_running
                min_x_new = 0
                min_y_orig = y_running
                min_y_new = 0
                max_x_orig = x_running + slicesize[1]
                max_x_new = slicesize[1]
                max_y_orig = y_running + slicesize[0]
                max_y_new = slicesize[0]
                try:
                    if (modality=='image'):
                        if (rescale):
                            img_to_save = np.zeros((slicesize[0], slicesize[1],slicesize[2]),dtype=np.float32)
                        else:
                            img_to_save = np.zeros((slicesize[0], slicesize[1], slicesize[2]), dtype=np.uint8)
                        img_to_save[:, :, 0] = img_to_save[:, :, 0] + thresh_img[0]
                        img_to_save[:, :, 1] = img_to_save[:, :, 1] + thresh_img[1]
                        img_to_save[:, :, 2] = img_to_save[:, :, 2] + thresh_img[2]
                    else:
                        img_to_save = np.zeros((slicesize[0], slicesize[1]), dtype=np.uint8)
                    if (x_running == 0):
                        max_x_orig = slicesize[1] - overlap
                        min_x_new = overlap

                    if (y_running == 0):
                        max_y_orig = slicesize[0] - overlap
                        min_y_new = overlap

                    if (max_y_orig > y):
                        max_y_orig = y
                        max_y_new = y - y_running

                    if (max_x_orig > x):
                        max_x_orig = x
                        max_x_new = x - x_running

                    #if ((x < slicesize[1]) & (x_running == 1)):
                    if (x < (slicesize[1]-overlap)):
                        max_x_new = max_x_new + overlap
                    #if ((y < slicesize[0]) & (y_running == 1)):
                    if (y < (slicesize[0]-overlap)):
                        max_y_new = max_y_new + overlap
                    if (modality=='image'):
                        img_to_save[min_y_new:max_y_new, min_x_new:max_x_new,:] = image[min_y_orig:max_y_orig, min_x_orig:max_x_orig,:]
                        if ((mode=='training') and (forunetorconvnet == 'CONVNET')):
                            mask_sum = mask[min_y_orig:max_y_orig, min_x_orig:max_x_orig].sum()
                            if (mask_sum > 0):
                                img_new.append(img_to_save)
                                scales_to_return.append(scale)
                        else:
                            img_new.append(img_to_save)
                            if (mode == 'eval'):
                                image_nr.append(i)
                    else:
                        img_to_save[min_y_new:max_y_new, min_x_new:max_x_new] = image[min_y_orig:max_y_orig, min_x_orig:max_x_orig]
                        img_new.append(img_to_save)
                except:
                    e=1
                y_running = y_running + slicesize[0] - 2 * overlap
                del img_to_save
            x_running = x_running + slicesize[1] - 2 * overlap
        #if (forunetorconvnet == 'CONVNET'):
        if (mode == 'training'):
            if (mode == 'UNET'):
                img_to_return.extend(img_new)
            else:
                #img_to_return.append(img_new)
                img_to_return.extend(img_new)
        else:
            img_to_return.extend(img_new)
            original_shape_to_return.append(image.shape)
        del img_new
    if (forunetorconvnet == 'UNET'):
        if (mode=='training'):
            return img_to_return
        else:
            return img_to_return, image_nr, original_shape_to_return
    else:
        if (mode == 'training'):
            return img_to_return, scales_to_return
        else:
            return img_to_return, image_nr

def reconstructImages(images=None,images_original_shape=None,scales=None,image_nr=None,overlap=0,mode='images'):
    img_to_return = []
    MEAN_NUCLEI_SIZE = np.load(os.path.join("tmp", "mean_nuclear_size.npy"))
    for i in tqdm(np.unique(image_nr)):
        images_tmp = np.array(images)[np.array(image_nr)==i]
        x_running = 0
        if (mode=='images'):
            img_to_save = np.zeros((images_original_shape[i][0],images_original_shape[i][1],images_original_shape[i][2]),dtype=np.float)
            [y, x, z] = img_to_save.shape
        else:
            img_to_save = np.zeros((images_original_shape[i][0],images_original_shape[i][1]),dtype=np.float)
            [y, x] = img_to_save.shape
        thresh_img = []
        slicesize = [INPUT_SHAPE[1],INPUT_SHAPE[2],INPUT_SHAPE[0]]

        running_ind = 0
        while (x_running < x):
            y_running = 0
            while (y_running < y):
                min_x_orig = x_running
                min_x_new = 0
                min_y_orig = y_running
                min_y_new = 0
                max_x_orig = x_running + slicesize[1]
                max_x_new = slicesize[1]
                max_y_orig = y_running + slicesize[0]
                max_y_new = slicesize[0]
                try:
                    if (x_running == 0):
                        max_x_orig = slicesize[1] - overlap
                        min_x_new = overlap

                    if (y_running == 0):
                        min_y_orig = 0
                        max_y_orig = slicesize[0] - overlap
                        min_y_new = overlap

                    if (max_y_orig > (y)):
                        max_y_orig = (y)
                        max_y_new = y - y_running

                    if (max_x_orig > x):
                        max_x_orig = x
                        max_x_new = x - x_running
                    if (x < (slicesize[1]-overlap)):
                        max_x_new = max_x_new + overlap
                    if (y < (slicesize[0]-overlap)):
                        max_y_new = max_y_new + overlap
                    if (mode == 'images'):
                        img_to_save[min_y_orig:max_y_orig, min_x_orig:max_x_orig, :] = images_tmp[running_ind][min_y_new :max_y_new,min_x_new:max_x_new,:]
                    else:
                        img_to_save[min_y_orig:max_y_orig, min_x_orig:max_x_orig] = images_tmp[running_ind][min_y_new:max_y_new,min_x_new:max_x_new]
                    running_ind = running_ind + 1
                except:
                    e = 1
                y_running = y_running + slicesize[0] - 2 * overlap
            x_running = x_running + slicesize[1] - 2 * overlap
        img_to_save = np.float32(ski_transform.resize(img_to_save, (int(img_to_save.shape[0] * (scales[i] / MEAN_NUCLEI_SIZE)),int(img_to_save.shape[1] * (scales[i] / MEAN_NUCLEI_SIZE))), mode='reflect'))
        img_to_return.append(img_to_save)
    return img_to_return

def rescale_mask (image,x_factor,y_factor):
    im_new = np.zeros([x_factor, y_factor], dtype=np.uint8)
    for i in range(1,image.max()+1):
        #img_tmp = ski_transform.resize(image==i, (x_factor,y_factor),mode='reflect')
        im_new = im_new + i * (ski_transform.resize(image==i, (x_factor,y_factor),mode='reflect')>0.5)
    return im_new

def getMeanMaskObjectSize(mask_files):
    total_sum = 0;
    for i in np.unique(mask_files):
	    if i>0:
	        total_sum = total_sum + (mask_files==(i)).sum()
    return (total_sum / (np.unique(mask_files).__len__()-1)).astype(np.int16)

def binSize(sizes):
    sizes = np.asarray(sizes, dtype=float)
    msk_to_ret = np.asarray(sizes, dtype=int)
    med = np.median(sizes)
    msk_to_ret[sizes < med] = 1
    msk_to_ret[sizes >= med] = 2
    return msk_to_ret

def write_to_list_h5py(filename, data):
    """

    :param filename:
    :param data:
    :return:
    """
    try:
        print(os.getcwd() + filename)
        h5_file = h5py.File(filename, 'w')
        h5_file.create_dataset('data', data=data)
        h5_file.close()
        print('File successfully saved in: {:s}'.format(filename))
    except IOError:
        raise IOError('Can not save data')

def write_to_h5py(filename, data):
    """

    :param filename:
    :param data:
    :return:
    """
    try:
        print(os.getcwd() + filename)
        h5_file = h5py.File(filename, 'w')
        for k, v in data.items():
            h5_file.create_dataset(k, data=v)
        h5_file.close()
        print('File successfully saved in: {:s}'.format(filename))
    except IOError:
        raise IOError('Can not save data')


def convert_masks_to_rle(masks, image_ids, save_file='sub-dsbowl2018-1_VISIOMICS.csv'):
    encodings = []
    sub = pd.DataFrame()
    id_tmp = []

    for i, mask in enumerate(tqdm(masks)):
        nlabels = np.unique(mask).max()
        if (nlabels == 0):
            encodings.append(rle_encode(np.uint8(mask == l)))
            id_tmp.append(image_ids[i])
        for l in range(1, nlabels+1):
            nucl_mask = np.uint8(mask == l)
            encodings.append(rle_encode(nucl_mask))
            id_tmp.append(image_ids[i])
            # rle_str = rle_to_string(rle)
            # decode_mask = rle_decode(rle_str, mask.shape, mask.dtype)

            # assert np.all(nucl_mask == decode_mask)

    # for i, m in enumerate(tqdm(masks)):
    #     if (m.max() == 0):
    #         encodings.append(rle_encoding(m==1))
    #         id_tmp.append(image_ids[i])
    #     for t in range(m.max()):
    #         encodings.append(rle_encoding(m==(t+1)))
    #         id_tmp.append(image_ids[i])
    # check output
    conv = lambda l: ' '.join(map(str, l))  # list -> string
    subject, img = 1, 1
    print('\n{},{},{}'.format(subject, img, conv(encodings[0])))

    # Create submission DataFrame and csv file
    #if save_file is not None:
    sub['ImageId'] = id_tmp
    sub['EncodedPixels'] = pd.Series(encodings).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(save_file, index=False)

    return encodings

def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# Used only for testing.
# This is copied from https://www.kaggle.com/paulorzp/run-length-encode-and-decode.
# Thanks to Paulo Pinto.
def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T
# def rle_encoding(x):
#     '''
#     x: numpy array of shape (height, width), 1 - mask, 0 - background
#     Returns run length as list
#     '''
#     dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if b > prev + 1:
#             run_lengths.extend((b + 1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths

def extract_ids (imagepath):
    for i in range(imagepath.__len__()):
        tmp = imagepath[i].split('\\')
        imagepath[i] = tmp[tmp.__len__()-1].split('.')[0]
    return imagepath
