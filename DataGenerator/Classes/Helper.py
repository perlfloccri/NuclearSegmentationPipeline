#
# Helper Tools for Image Processing and SVG Transformation
# Written by Florian Kromp
# Children's Cancer Research Institute
# florian.kromp@ccri.at
# Last Update 16.01.2019

import sys
from Classes.Config import Config
from Classes.Image import AnnotatedImage, ArtificialAnnotatedImage
sys.path.append(r'D:\DeepLearning\Kaggle\TileImages')
#from tools import rescaleAndTile,getMeanMaskObjectSize
from Classes.Image import AnnotatedImage
import tifffile
import numpy as np
from tqdm import tqdm
import os
import skimage.transform as ski_transform
from skimage import filters
import scipy.misc
import matplotlib.pyplot as plt
import glob
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image,ImageEnhance
import cv2
import skimage
from random import randint
from skimage import transform as trf
from random import uniform
INPUT_SHAPE = [1,256,256]
from skimage.measure import label
from skimage import measure
import xml.etree.ElementTree as ET
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import scipy.fftpack as fp
import math
from skimage.morphology import disk, dilation

class Tools:

    MEAN_NUCLEI_SIZE = 0.2

    def getLocalDataPath(self,path,content):
        config = Config
        if (content == 1): # Image
            erg = str.replace(path,'/var/www/TisQuant/data/',config.local_data_path)
        elif (content==2):
            erg = str.replace(path,'/var/www/TisQuant/data/automaticresult/',config.local_data_path + 'automaticresult\\')
        elif (content==3):
            erg = str.replace(path,'/var/www/TisQuant/data/groundtruth/', config.local_data_path + 'groundtruth\\')
        elif (content==2):
            erg = str.replace(path,'/var/www/TisQuant/data/database/',config.local_data_path + 'database\\')
        else:
            erg=path
        return erg

    def createAndSaveTiles(self,annotated_nuclei,config, img_prefix='Img_',mask_prefix='Mask_'):
        print(img_prefix)
        print(mask_prefix)
        images = []
        masks = []
        path_to_img = []
        if config.mode == 'test':
            diagnosis = []
        for i in range(0,annotated_nuclei.images.__len__()):
            images.append(annotated_nuclei.images[i].getRaw())
            masks.append(annotated_nuclei.images[i].getMask())
            path_to_img.append(annotated_nuclei.path_to_imgs[i])
        # Get scales from masks
        print("Calculate mean object size ...")
        scales_for_conv = self.getNormalizedScales(annotated_nuclei.images)
        print(scales_for_conv)

            # Rescale and Tile
        print("Rescale and tile images and masks to " + config.outputFolder + "...")
        [images,masks,img_index,tile_index,tile_scales] = self.rescaleAndTile(images=images,masks=masks,scales=scales_for_conv,overlap = config.overlap,rescale=config.scale,mode=config.mode,path_to_img=path_to_img)

        # Create artificial dataset
        if (config.diagnosis.__len__() > 1):
            img_name = 'combined'
        else:
            img_name = config.diagnosis[0]

        print("Save tiled dataset ...")
        for i in range(0, images.__len__()):
            scipy.misc.toimage(images[i], cmin=0.0, cmax=1.0).save(config.outputFolder + '\\images\\' + img_prefix + img_name + '_' + str(i) + '.jpg')
            tifffile.imsave(config.outputFolder + '\\masks\\' + mask_prefix + img_name + '_' + str(i) + '.tif',(masks[i]).astype(np.uint8))
            if config.mode == 'test':
                with open(config.resultsfile, 'a') as f:
                    #f.write(config.outputFolder + ',' + str(img_index[i]) + ',' + str(tile_index[i]) + "\n")
                    f.write(img_index[i] + ',' + str(tile_scales[i]) + ',' + str(tile_index[i]) + "\n")

    def createAndSaveTilesForSampleSegmentation(self,annotated_nuclei,config,scale):
        images = []
        path_to_img = []
        scales_for_conv = []
        for i in range(0,annotated_nuclei.images.__len__()):
            images.append(annotated_nuclei.images[i].getRaw())
            path_to_img.append(annotated_nuclei.path_to_imgs[i])
            scales_for_conv.append(scale)

            # Rescale and Tile
        print("Rescale and tile images and masks to " + config.outputFolder + "...")
        [images,img_index,tile_index,tile_scales] = self.rescaleAndTileForSampleSegmentation(images=images,scales=scales_for_conv,overlap = config.overlap,rescale=config.scale,mode=config.mode,path_to_img=path_to_img)
        print(images.__len__())
        print(img_index.__len__())
        print(tile_index.__len__())
        print(tile_scales.__len__())
        # Create artificial dataset

        print("Save tiled dataset ...")
        print(config.outputFolder)
        print (config.outputFolder + '\\' + os.path.basename(img_index[i]).replace('.'+os.path.basename(img_index[i]).split('.')[1],'_' + self.getNumeration(i) + '.jpg'))
        for i in range(0, images.__len__()):
            scipy.misc.toimage(images[i], cmin=0.0, cmax=1.0).save(config.outputFolder + '\\' + os.path.basename(img_index[i]).replace('.'+os.path.basename(img_index[i]).split('.')[1],'_' + self.getNumeration(i) + '.jpg'))
            scipy.misc.toimage(np.zeros_like(images[i],dtype=np.uint8)+1).save(config.outputFolder.replace('images','masks') + '\\' + os.path.basename(img_index[i]).replace('.'+os.path.basename(img_index[i]).split('.')[1],'_' + self.getNumeration(i) + '.tif'))
            #tifffile.imsave(config.outputFolder + '\\' + os.path.basename(img_index[i]).replace('.'+os.path.basename(img_index[i]).split('.')[1],'_' + self.getNumeration(i) + '.tif'),images[i])
            with open(config.resultsfile, 'a') as f:
                f.write(img_index[i] + ',' + str(tile_scales[i]) + ',' + str(tile_index[i]) + "\n")

    def rescaleAndTileForSampleSegmentation (self,images=None,scales=None,rescale=True,overlap=20,mode=None,path_to_img=None):
        img_to_return = []
        pathes_to_return = []
        img_path_to_return = []
        index_to_return = []
        tile_scales = []
        nr_images = images.__len__()
        print("Rescale ...")
        print(rescale)
        for i in tqdm(range(nr_images)):
            if (rescale):
                image = self.rescale_image(images[i],(scales[i],scales[i]))
            else:
                image = images[i]
            x_running = 0
            img_new = []
            mask_new = []
            thresh_img = []
            slicesize = [INPUT_SHAPE[1],INPUT_SHAPE[2],INPUT_SHAPE[0]]
            thresh_img.append((np.mean(image[np.where(image < filters.threshold_otsu(image))])))
            [y, x] = image.shape
            running_index = 0
            while (x_running <= (x - overlap)):

                y_running = 0
                while (y_running <= (y - overlap)):
                    min_x_orig = x_running
                    min_x_new = 0
                    min_y_orig = y_running
                    min_y_new = 0
                    max_x_orig = x_running + slicesize[1]
                    max_x_new = slicesize[1]
                    max_y_orig = y_running + slicesize[0]
                    max_y_new = slicesize[0]
                    try:
                        img_to_save = np.zeros((slicesize[0], slicesize[1]),dtype=np.float32)
                        img_to_save = img_to_save + thresh_img[0]
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
                        if (x < (slicesize[1]-overlap)):
                            max_x_new = max_x_new + overlap
                        if (y < (slicesize[0]-overlap)):
                            max_y_new = max_y_new + overlap
                        img_to_save[min_y_new:max_y_new, min_x_new:max_x_new] = image[min_y_orig:max_y_orig, min_x_orig:max_x_orig]
                        img_new.append(img_to_save)
                        try: # change and check which programm calls the function
                            img_path_to_return.append(path_to_img[i])
                            index_to_return.append(running_index)
                            tile_scales.append(scales[i])
                        except:
                            print("No pathes given")
                        running_index = running_index+1
                    except:
                        print('failed to tile....')
                        input("Press Enter to continue...")
                    y_running = y_running + slicesize[0] - 2 * overlap
                    del img_to_save
                x_running = x_running + slicesize[1] - 2 * overlap

            img_to_return.extend(img_new)
            del img_new
        return img_to_return,img_path_to_return,index_to_return,tile_scales

    def getNumeration(self,i):
        return "{0:0>4}".format(i)

    def visualize_frequencies(self,annotated_images):
        number = annotated_images.__len__()
        plt.figure(1)
        for index,image in enumerate(annotated_images):
            plt.subplot(2,number,index+1)
            F1 = fp.fft2(image.astype(float))
            F2 = fp.fftshift(F1)
            plt.imshow(image, cmap='gray');
            plt.axis('off')
            plt.subplot(2, number, index + number + 1)
            plt.imshow((20 * np.log10(0.1 + F2)).astype(int), cmap=plt.cm.gray)
            plt.axis('off')

    def createPix2pixDataset(self,annotated_nuclei,config,n_freq=30,tissue=None):
        images = []
        masks = []

        for i in range(0,annotated_nuclei.images.__len__()):
            images.append(annotated_nuclei.images[i].getRaw())
            masks.append(annotated_nuclei.images[i].getMask())

        # Get scales from masks
        print("Calculate mean object size ...")
        #scales_for_conv = self.getNormalizedScales(masks)
        scales_for_conv = self.getNormalizedScales(annotated_nuclei.images)

        # Rescale and Tile
        print("Rescale and tile images and masks ...")
        [images,masks,t,t,t] = self.rescaleAndTile(images=images,masks=masks,scales=scales_for_conv,overlap = 0,rescale=config.scale,usePartial=False)

        # Create artificial dataset
        if (config.diagnosis.__len__() > 1):
            img_name = 'combined'
        else:
            img_name = config.diagnosis[0]

        print("Create artificial dataset ...")
        for i in range(0, images.__len__() - 1):

            # calculate background
            tmp_image = images[i]
            tmp_mask = masks[i]
            kernel = np.ones((15, 15), np.uint8)
            bg = cv2.erode((tmp_mask == 0).astype(np.uint8), kernel, iterations=1)
            bg = np.sort(tmp_image[np.where(bg > 0)])

            img_nat = AnnotatedImage();
            img_nat.createWithArguments(images[i],masks[i])
            img_art = ArtificialAnnotatedImage
            img_art = img_art.transformToArtificialImage(image=img_nat,useBorderObjects=config.useBorderObjects,background=bg)
            img_art_beforefiltering = AnnotatedImage()
            img_art_beforefiltering.createWithArguments(img_art.getRaw(),img_art.getMask())

            #borders = cv2.dilate((cv2.Laplacian(tmp_mask,cv2.CV_64F)>0).astype(np.uint16), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            #original_raw = img_art_beforefiltering.getRaw()

            #img_art.filterLowFrequencies(n=n_freq)

            #pixels_to_change = np.where(borders>0)
            #original_raw_new = np.copy(original_raw)
            #original_raw_new[pixels_to_change] = img_art.getRaw()[pixels_to_change]
            #if not (tissue == 'Ganglioneuroma'):
            #    img_art.raw = original_raw_new.astype(img_art.raw.dtype)

            #self.visualize_frequencies([img_nat.getRaw(),img_art_beforefiltering.getRaw(),img_art.filterLowFrequencies(img_art_beforefiltering.getRaw(),n=20),img_art.filterLowFrequencies(img_art_beforefiltering.getRaw(),n=30),img_art.filterLowFrequencies(img_art_beforefiltering.getRaw(),n=40),img_art.getRaw()])
            #plt.show(block=False)
            img_combined = np.zeros((images[0].shape[0], images[0].shape[1] * 2),np.float32)
            img_combined[:,0:INPUT_SHAPE[1]] = img_nat.getRaw()
            img_combined[:, INPUT_SHAPE[1]:INPUT_SHAPE[1]*2] = img_art.getRaw()
            plt.imshow(img_combined,cmap='gray')
            img_to_sav = np.zeros((img_combined.shape[0],img_combined.shape[1],3),np.float32)
            img_to_sav[:, :, 0] = img_combined
            img_to_sav[:, :, 1] = img_combined
            img_to_sav[:, :, 2] = img_combined
            #scipy.misc.toimage(img_to_sav, cmin=0.0, cmax=1.0).save(config.outputPath + config.outputFolder + '\\Img_' + str(i) + '.jpg')

            scipy.misc.toimage(img_to_sav, cmin=0.0, cmax=1.0).save(config.outputFolder + '\\Img_' + img_name + '_' + str(i) + '.jpg')
            e=1
            #tifffile.imsave('D:\\DeepLearning\\DataGenerator\\Dataset\\Natural\\Natural_img_' + str(i) + '.tif',(annotated_nuclei.images[i].getRaw() * 255.0).astype(np.uint8))
            #img = ArtificialAnnotatedImage.transformToArtificialImage(annotated_nuclei.images[i])
            #tifffile.imsave('D:\\DeepLearning\\DataGenerator\\Dataset\\Artificial\\Artificial_img_' + str(i) + '.tif',(img.getRaw() * 255.0).astype(np.uint8))

    def rescaleAndTile (self,images=None,masks=None,scales=None,rescale=True,overlap=20,mode=None,path_to_img=None,usePartial=True):
        img_to_return = []
        mask_to_return = []
        img_path_to_return = []
        index_to_return = []
        tile_scales = []
        nr_images = images.__len__()

        for i in tqdm(range(nr_images)):
            if (rescale):
                #image  = np.float32(ski_transform.resize(images[i], (int(images[i].shape[0] * 1 / (scales[i] / MEAN_NUCLEI_SIZE)), int(images[i].shape[1] * 1 / (scales[i] / MEAN_NUCLEI_SIZE))), mode='reflect'))
                #mask = self.rescale_mask(masks[i],int(masks[i].shape[0] * 1 / (scales[i] / self.MEAN_NUCLEI_SIZE)), int(masks[i].shape[1] * 1 / (scales[i] / self.MEAN_NUCLEI_SIZE)))
                image = self.rescale_image(images[i],(scales[i],scales[i]))
                mask = self.rescale_mask(masks[i], (scales[i],scales[i]),make_labels=True)
            else:
                image = images[i]
                mask = masks[i]
            x_running = 0
            img_new = []
            mask_new = []
            thresh_img = []
            slicesize = [INPUT_SHAPE[1],INPUT_SHAPE[2],INPUT_SHAPE[0]]
            thresh_img.append((np.mean(image[np.where(image < filters.threshold_otsu(image))])))
            [y, x] = image.shape
            running_index = 0
            while (x_running <= (x - overlap)):

                y_running = 0
                while (y_running <= (y - overlap)):
                    min_x_orig = x_running
                    min_x_new = 0
                    min_y_orig = y_running
                    min_y_new = 0
                    max_x_orig = x_running + slicesize[1]
                    max_x_new = slicesize[1]
                    max_y_orig = y_running + slicesize[0]
                    max_y_new = slicesize[0]
                    try:
                        img_to_save = np.zeros((slicesize[0], slicesize[1]),dtype=np.float32)
                        mask_to_save = np.zeros((slicesize[0], slicesize[1]), dtype=np.uint8)
                        img_to_save = img_to_save + thresh_img[0]
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
                        if (x < (slicesize[1]-overlap)):
                            max_x_new = max_x_new + overlap
                        if (y < (slicesize[0]-overlap)):
                            max_y_new = max_y_new + overlap
                        img_to_save[min_y_new:max_y_new, min_x_new:max_x_new] = image[min_y_orig:max_y_orig, min_x_orig:max_x_orig]
                        mask_to_save[min_y_new:max_y_new, min_x_new:max_x_new] = mask[min_y_orig:max_y_orig, min_x_orig:max_x_orig]
                        if (((mask_to_save.max()>0) & ((mask_to_save>0).sum() > 100)) | (mode == 'test')):
                            if usePartial or ((usePartial==False) and ((max_y_orig-min_y_orig)>=(slicesize[0]-1)) and ((max_x_orig-min_x_orig)>=(slicesize[1]-1))):
                                img_new.append(img_to_save)
                                mask_new.append(mask_to_save)
                                try: # change and check which programm calls the function
                                    img_path_to_return.append(path_to_img[i])
                                    index_to_return.append(running_index)
                                    tile_scales.append(scales[i])
                                except:
                                    print("No pathes given")
                                running_index = running_index+1
                    except:
                        print('failed to tile....')
                        input("Press Enter to continue...")
                    y_running = y_running + slicesize[0] - 2 * overlap
                    del img_to_save
                x_running = x_running + slicesize[1] - 2 * overlap

            img_to_return.extend(img_new)
            mask_to_return.extend(mask_new)
            del img_new
            del mask_new
        return img_to_return,mask_to_return,img_path_to_return,index_to_return,tile_scales

    def reconstruct_images(self,images=None,predictions=None,scales=None,rescale=True,overlap=20,config=None,label_output=False, dilate_objects=False):
        img_to_return = []
        mask_to_return = []

        nr_images = images.__len__()
        running_ind = 0
        for i in tqdm(range(nr_images)):
            if (rescale):
                image = self.rescale_image(images[i],(scales[i],scales[i]))
            else:
                image = images[i]
            x_running = 0
            img_new = []
            mask_new = []
            thresh_img = []
            slicesize = [INPUT_SHAPE[1],INPUT_SHAPE[2],INPUT_SHAPE[0]]
            [y, x] = image.shape
            img_to_save = np.zeros((y, x), dtype=np.float32)
            mask_to_save = np.zeros((y, x), dtype=np.float32)
            while (x_running <= (x-overlap)):

                y_running = 0
                while (y_running <= (y-overlap)):
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
                            max_y_orig = slicesize[0] - overlap
                            min_y_new = overlap
                        if (max_y_orig > y):
                            max_y_orig = y
                            max_y_new = y - y_running
                        if (max_x_orig > x):
                            max_x_orig = x
                            max_x_new = x - x_running
                        if (x < (slicesize[1]-overlap)):
                            max_x_new = max_x_new + overlap
                        if (y < (slicesize[0]-overlap)):
                            max_y_new = max_y_new + overlap
                        # New: only use half of the overlap
                        if (y_running != 0):
                            min_y_new = min_y_new + int(overlap/2)
                            min_y_orig = min_y_orig + int(overlap/2)
                        if (x_running != 0):
                            min_x_new = min_x_new + int(overlap/2)
                            min_x_orig = min_x_orig + int(overlap/2)
                        img_to_save[min_y_orig:max_y_orig, min_x_orig:max_x_orig] = predictions[running_ind][min_y_new:max_y_new, min_x_new:max_x_new]
                        mask_to_save[min_y_orig:max_y_orig, min_x_orig:max_x_orig] = predictions[running_ind][min_y_new:max_y_new, min_x_new:max_x_new]>0.3
                        running_ind = running_ind + 1
                    except:
                        e=1
                    y_running = y_running + slicesize[0] - 2 * overlap
                x_running = x_running + slicesize[1] - 2 * overlap
            if (rescale):
                img_to_save = self.upscale_image(img_to_save,(images[i].shape[0],images[i].shape[1]),config=config)
                mask_to_save = self.upscale_mask(mask_to_save,(images[i].shape[0],images[i].shape[1]))
            img_to_return.append(img_to_save)
            if label_output:
                #print("Label!!")
                mask_labeled = label(self.postprocess_mask(mask_to_save).astype(np.uint8))
                if dilate_objects:
                    for i in np.unique(mask_labeled):
                        if i>0:
                            #print("Dilate object!")
                            tmp = mask_labeled == i
                            tmp = dilation(tmp,disk(dilate_objects))
                            mask_labeled[np.where(tmp>0)] = 0
                            mask_labeled += tmp*i
                mask_to_return.append(mask_labeled)
            else:
                mask_tmp = self.postprocess_mask(mask_to_save)
                if dilate_objects:
                    for i in np.unique(mask_labeled):
                        if i>0:
                            tmp = mask_tmp == i
                            tmp = dilation(tmp,disk(dilate_objects))
                            mask_tmp[np.where(tmp>0)] = 0
                            mask_tmp += tmp
                mask_to_return.append(mask_tmp)
            #mask_to_return.append(self.postprocess_mask(mask_to_save))
            del img_to_save
        return img_to_return, mask_to_return

    def postprocess_mask(self,mask,threshold=3):
        mask = label(mask)
        for i in np.unique(mask):
            if i>0:
                if ((mask==i).sum() < threshold):
                    mask[mask==i] = 0
        return mask>0

    def rescale_mask(self, image, scale,make_labels=None):
        x_factor = int(image.shape[0] * 1 / (scale[0] / self.MEAN_NUCLEI_SIZE))
        y_factor = int(image.shape[1] * 1 / (scale[0] / self.MEAN_NUCLEI_SIZE))
        im_new = np.zeros([x_factor, y_factor], dtype=np.uint8)
        for i in tqdm(range(1,image.max()+1)):
            if make_labels:
                im_new = im_new + i * (ski_transform.resize(image==i, (x_factor,y_factor),mode='reflect')>0.5)
            else:
                im_new = im_new + (ski_transform.resize(image==i, (x_factor,y_factor),mode='reflect')>0.5)
        return im_new

    def upscale_mask(self,image,scale):
        image = scipy.ndimage.label(image)[0]
        im_new = np.zeros([scale[0], scale[1]], dtype=np.float32)
        for i in tqdm(range(1,image.max()+1)):
            im_new = im_new + (ski_transform.resize(image==i, (scale[0],scale[1]),mode='reflect')>0.5)
        return im_new

    #def rescale_image(self,image,x_factor,y_factor):
    def rescale_image(self, image, scale):
        x_factor = int(image.shape[0] * 1 / (scale[0] / self.MEAN_NUCLEI_SIZE))
        y_factor = int(image.shape[1] * 1 / (scale[0] / self.MEAN_NUCLEI_SIZE))
        return np.float32(ski_transform.resize(image, (x_factor,y_factor), mode='reflect'))

    def upscale_image(self, image, scale,config=None):
        if config.net == 'maskrcnn':
            return np.float32(ski_transform.resize(image>0, (scale[0], scale[1]), mode='reflect'))>0
        else:
            return np.float32(ski_transform.resize(image, (scale[0],scale[1]), mode='reflect'))

    def getMeanAndStdSizeOfMask(self,mask):
        scales = []
        for i in np.unique(mask):
            if i>0:
                scales.append((mask==i).sum())
        return np.mean(scales), np.std(scales)

    def getNormalizedScales(self,masks):
        scales = []
        for mask in tqdm(masks):
            #scales.append(int(self.getMeanMaskObjectSize(mask)))
            scales.append(int(mask.getMeanMaskObjectSize()))

        # Scale groundtruth to be between 0 and 1
        print("Scale grountruth to be between 0 and 1 ...")
        max_nucl_size = 170
        scales_for_conv = [float(x) / max_nucl_size for x in scales]
        for i in range(scales_for_conv.__len__()):
            if (scales_for_conv[i] > 1):
                scales_for_conv[i] = 1
        return scales_for_conv

    def createTisquantLikeDataset(self,path,output):
        print(path)
        image_list = glob.glob(os.path.join(path,'results','normal','images','*-outputs.png'))
        mask_list = glob.glob(os.path.join(path,'ArtToNat','running','normal','masks','*.tif'))
        print(image_list)
        print(mask_list)

    def elastic_transformations(self,alpha, sigma, image_shape, rng=np.random.RandomState(42),
                                interpolation_order=1):
        """Returns a function to elastically transform multiple images."""
        # Good values for:
        #   alpha: 2000
        #   sigma: between 40 and 60

        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        # image_shape = images[0].shape

        # Make random fields
        # random.seed(nbr_seed)
        dx = rng.uniform(-1, 1, image_shape) * alpha
        dy = rng.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))

        def _elastic_transform_2D(image):
            # Distort meshgrid indices
            distorted_indices = (y + sdy).reshape(-1, 1), \
                                (x + sdx).reshape(-1, 1)

            # Map cooordinates from image to distorted index set
            transformed_image = map_coordinates(image, distorted_indices, mode='reflect',
                                                  order=interpolation_order).reshape(image_shape)

            return transformed_image

        return _elastic_transform_2D

    def enhanceImage(self,img,flip_left_right=None,flip_up_down=None,deform=None):
        img_list = []
        img_list.append(img)

        try:
            xrange
        except NameError:
            xrange = range
        # flipping
        if flip_left_right:
            for i in xrange(0,img_list.__len__()):
                x = img_list[i].getRaw()
                y = img_list[i].getMask()
                x = np.fliplr(x)
                y = np.fliplr(y)
                img_new = AnnotatedImage()
                img_new.createWithArguments(x,y)
                img_list.append(img_new)
        if flip_up_down:
            for i in xrange(0, img_list.__len__()):
                x = img_list[i].getRaw()
                y = img_list[i].getMask()
                x = np.flipud(x)
                y = np.flipud(y)
                img_new = AnnotatedImage()
                img_new.createWithArguments(x,y)
                img_list.append(img_new)
        if deform:
            for i in xrange(0, img_list.__len__()):
                x = img_list[i].getRaw()
                y = img_list[i].getMask()
                for t in xrange(0,5):
                    def_func = self.elastic_transformations(2000, 60, x.shape)
                    x = def_func(x)
                    y_new = np.zeros((y.shape[0],y.shape[1]),dtype=np.uint16)
                    for z in xrange(0,y.max()+1):
                        y_tmp = def_func((y==z)*255)
                        y_new = y_new + (z * (y_tmp==255)).astype(np.uint16)
                    y=y_new
                    img_new = AnnotatedImage()
                    img_new.createWithArguments(x,y)
                    img_list.append(img_new)

        return img_list

    def arbitraryEnhance(self,annotated_image):
        x = annotated_image.getRaw()
        y = annotated_image.getMask()
        try:
            xrange
        except NameError:
            xrange = range

        if randint(0,1):  # flip horizontally
            x = np.fliplr(x)
            y = np.fliplr(y)
        if randint(0,1):  # flipping vertically
            x = np.flipud(x)
            y = np.flipud(y)
        if 0:#randint(0,1):  # deform
            def_func = self.elastic_transformations(2000, 60, x.shape)
            x = def_func(x)
            y_new = np.zeros((y.shape[0],y.shape[1]),dtype=np.uint16)
            for z in xrange(0,y.max()+1):
                y_tmp = def_func((y==z)*255)
                y_new = y_new + (z * (y_tmp==255)).astype(np.uint16)
            y=y_new
        if randint(0,1): # rotate
            x_rot = np.zeros_like(x)
            y_rot = np.zeros_like(y)
            rot_angle = np.random.randint(-90, 90)
            x = trf.rotate(x, rot_angle)
            y = trf.rotate(y.squeeze(), rot_angle, order=0)
        if 0: #randint(0, 1): # enhance brightness
            nucl_pixels = x * y
            pixels = np.where(nucl_pixels > 0)
            x[x < 0] = 0.0
            x[x > 1.0] = 1.0
            if ((nucl_pixels[pixels].mean() > 0.2) and (nucl_pixels[pixels].mean() < 0.5)):
                x[pixels] += uniform(0,0.3)
            elif (nucl_pixels[pixels].mean() < 0.8):
                x[pixels] -= uniform(0, 0.3)
            x[x<0] = 0
            x[x > 1] = 1.0
        if randint(0,1): # gaussian
            x = x * 255.0
            x = x + np.random.normal(0, 2, [x.shape[0], x.shape[1]])
            x[x<0] = 0
            x[x > 255] = 255
            x = x / 255.0
        if randint(0,1): #blur
            x = x * 255.0
            kernel_size = np.random.randint(1,3)
            if (kernel_size%2 == 0):
                kernel_size = kernel_size+1;
            x = cv2.GaussianBlur(x,(kernel_size,kernel_size),0)
            x[x<0] = 0
            x[x > 255] = 255
            x = x / 255.0
        if randint(0,1):
            pixels = np.where(y > 0)
            range_scale = uniform(0.8,1.2)
            x = ski_transform.resize(x, (int(x.shape[0] * range_scale), int(x.shape[1] * range_scale)), mode='reflect')
            y = (ski_transform.resize(y, (int(y.shape[0] * range_scale), int(y.shape[1] * range_scale)), mode='reflect')>0.5)
        img_new = AnnotatedImage()
        img_new.createWithArguments(x,y)
        return img_new

class SVGTools:
    svg_str = ''
    height=None
    width=None
    samplingrate = None

    def __init__(self,samplingrate=10):
        self.samplingrate = int(samplingrate)

    def openSVG(self,height,width):
        self.height=height
        self.width=width
        self.svg_str = '<svg height="' + str(height) + '" width="' + str(width) + '" x="0px" y="0px">\n'

    def closeSVG(self):
        self.svg_str = self.svg_str + '</svg>\n'

    def writeToPath(self,path):
        file = open(path,'w')
        file.write(self.svg_str)
        file.close()

    def addRawImage(self,name=None,img_path=None):
        self.svg_str += '<g id="' + name + '">\n'
        self.svg_str = self.svg_str + '\t<image xlink:href = "' + img_path + '" x = "0" y = "0" height = "' + str(self.height) + 'px" width = "' + str(self.width) + 'px" />'
        self.svg_str += "\n</g>\n"

    def addMaskLayer(self,mask,name,color,opacity):
        svg_str = ''
        contours = []
        for i in range (1,mask.max()+1):
            if ((mask==i).sum() > 0):
                contours.append(measure.find_contours(mask==i, 0.5))
        svg_str = '<g id="' + name + '" opacity="' + str(opacity) + '">'
        for index, contour in enumerate(contours):
            svg_str = svg_str + '\t<polygon fill="' + color + '" stroke="#800080" points="'
            for i in range(0,contour[0].__len__(),self.samplingrate):
                svg_str = svg_str + str(int(contour[0][i, 1])) + ',' + str(int(contour[0][i, 0])) + ' '
            #svg_str = svg_str +'" style="fill:lime;stroke:purple;stroke-width:1" />\n'
            svg_str = svg_str + '" style="stroke:purple;stroke-width:1" />\n'
        self.svg_str = self.svg_str + svg_str + '</g>\n'

    def getSVGMask(self,img_path=None):
        contours = []
        for i in range (1,self.mask.max()):
            if ((self.mask==i).sum() > 0):
                contours.append(measure.find_contours(self.mask==i, 0.5))
        #contours = measure.find_contours(self.mask, 1)
        svg_str = ''
        svg_str = svg_str + '<svg height="' + str(self.mask.shape[0]) + '" width="' + str(self.mask.shape[1]) + '">\n'
        for index, contour in enumerate(contours):
            svg_str = svg_str + '\t<polygon points="'
            for i in range(0,contour[0].__len__(),5):
                svg_str = svg_str + str(int(contour[0][i, 1])) + ',' + str(int(contour[0][i, 0])) + ' '
            svg_str = svg_str +'" style="fill:lime;stroke:purple;stroke-width:1" />\n'
        if img_path:
            svg_str = svg_str + '<image xlink:href = "' + img_path + '" x = "0" y = "0" height = "' + str(self.mask.shape[0]) + 'px" width = "' + str(self.mask.shape[1]) + 'px" />'
        svg_str = svg_str + '</svg>\n'
        return svg_str

    def transformSVGToMask(self,path):
        print(path)
        tree = ET.parse(path)
        root = tree.getroot()
        #img = np.zeros((root.get("width"),root.get("height")),astype=np.uint8)
        image = Image.new("L", (int(root.get("width").split('px')[0]),int(root.get("height").split('px')[0])))
        draw = ImageDraw.Draw(image)

        for i in range(0,root[3].getchildren().__len__()):
            points = []
            try:
                points_tmp = root[3].getchildren()[i].get("points").split(' ')
                for t in points_tmp:
                    try:
                        x,y = t.split(',')
                        points.append((round(float(x)),round(float(y))))
                    except:
                        None
            except:
                None
            if points:
                draw.polygon((points), fill=i+1)
        img = np.array(image)
        return img

    def transformSVGToMaskNew(self,path):
        print(path)
        tree = ET.parse(path)
        root = tree.getroot()
        img = np.zeros((int(root.get("height").split('px')[0]),int(root.get("width").split('px')[0])),dtype=np.uint16)

        labels = np.zeros((root[3].getchildren().__len__()))
        for i in range(0,root[3].getchildren().__len__()):
            labels[i] = i+1
        np.random.shuffle(labels)
        for i in range(0,root[3].getchildren().__len__()):
            image = Image.new("L", (int(root.get("width").split('px')[0]), int(root.get("height").split('px')[0])))
            draw = ImageDraw.Draw(image)
            points = []
            try:
                points_tmp = root[3].getchildren()[i].get("points").split(' ')
                for t in points_tmp:
                    try:
                        x,y = t.split(',')
                        #points.append((round(float(x.replace('.',''))),round(float(y.replace('.','')))))
                        points.append((round(float(x)), round(float(y))))
                    except:
                        None
            except:
                None
            if points:
                draw.polygon((points), fill=i+1)
            img_tmp = np.array(image)
            img[np.where((img_tmp>0).astype(np.uint8) == 1)] = 0
            img = img + (img_tmp>0).astype(np.uint16) * labels[i]
        return img.astype(np.uint16)