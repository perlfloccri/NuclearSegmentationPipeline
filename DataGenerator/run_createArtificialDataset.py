from Classes.Config import Config
from Classes.Helper import Tools
from Classes.Image import AnnotatedImage,AnnotatedObjectSet, ArtificialAnnotatedImage
from matplotlib import pyplot as plt
import scipy.misc
import random
import numpy as np
from tifffile import tifffile
import argparse
import glob
from random import uniform
import os
from tqdm import tqdm
import cv2
from random import randint
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--tissue', help='select tissue to train.', default=None)
    parser.add_argument('--inputFolder', help='Select input folder.', default=None)
    parser.add_argument('--outputFolder', help='select output folder', default=None)
    parser.add_argument('--nr_images', help='select number of images to create', default=None)
    parser.add_argument('--overlapProbability', help='select overlapProbability', default=None)
    parser.add_argument('--scale', help='select output folder', default=None)
    parser.add_argument('--img_prefix', help='select output folder', default='Img_')
    parser.add_argument('--mask_prefix', help='select output folder', default='Mask_')
    #random.seed(13431)
    args = parser.parse_args()
    config = Config
    if args.tissue:
        config.diagnosis = [args.tissue]

    if args.outputFolder:
        config.outputFolder = args.outputFolder

    if args.overlapProbability:
        args.overlapProbability = float(args.overlapProbability)
    else:
        args.overlapProbability = 0.5

    if args.tissue == 'Ganglioneuroma':
        n_freq = 20#15
    else:
        n_freq = 30

    if args.scale == '1':
        config.scale=True

    print(config.diagnosis)
    tools = Tools()

    annotated_nuclei =[]
    annotated_images = []
    ids_images = glob.glob(os.path.join(args.inputFolder,config.diagnosis[0],'images','*.tif'))
    ids_masks = glob.glob(os.path.join(args.inputFolder, config.diagnosis[0], 'masks', '*.tif'))

    for index, elem in enumerate(ids_images):
        test = AnnotatedImage()
        test.readFromPath(ids_images[index], ids_masks[index],type='uint16')
        annotated_images.append(test)

    # Create artificial new dataset
    scales = tools.getNormalizedScales(annotated_images)
    running = 0
    for index,img in enumerate(annotated_images):
        test = AnnotatedImage()
        annotated_nuclei.append(AnnotatedObjectSet())
        if config.scale:
            test.createWithArguments(tools.rescale_image(img.getRaw(),(scales[index],scales[index])),tools.rescale_mask(img.getMask(),(scales[index],scales[index]), make_labels=True))
        else:
            test.createWithArguments(img.getRaw(),img.getMask())
        annotated_nuclei[running].addObjectImage(test, useBorderObjects=config.useBorderObjects, tissue=args.tissue, scale=Config.scale)
        running += 1
        del test
    if config.scale == 0:
            if args.tissue == 'Ganglioneuroma':
                possible_numbers = [9, 16, 25, 36, 49]
            else:
                possible_numbers = [4, 4, 9]
    else:
        possible_numbers = [9,16,25,36,49]

    # How many images?
    if not args.nr_images:
        args.nr_images=10
    else:
        args.nr_images=int(args.nr_images)

    for t in tqdm(range(0,args.nr_images)):
        nr_img = random.randint(0,annotated_nuclei.__len__()-1)
        # Create artificial image
        number_nuclei = random.randint(0, possible_numbers.__len__()-1)

        # calculate Background
        tmp_image = annotated_nuclei[nr_img].images[0].getRaw()
        tmp_mask = annotated_nuclei[nr_img].images[0].getMask()
        kernel = np.ones((15, 15), np.uint8)
        bg = cv2.erode((tmp_mask == 0).astype(np.uint8), kernel, iterations=1)
        bg = np.sort(tmp_image[np.where(bg>0)])
        img = ArtificialAnnotatedImage(width=256,height=256,number_nuclei=possible_numbers[number_nuclei],probabilityOverlap=args.overlapProbability,background=bg)
        total_added = 0
        for i in range(0,possible_numbers[number_nuclei]):
            test = annotated_nuclei[nr_img].returnArbitraryObject()
            if (randint(0,1)):
                test = tools.arbitraryEnhance(test)
                total_added += img.addImageAtGridPosition(test)
        if (total_added > 0):
            shape_y = img.getRaw().shape[0]
            shape_x = img.getRaw().shape[1]
            img_new = np.zeros((shape_y,shape_x*2,3),dtype=np.float32)
            img_new[:,0:shape_x,0] = img_new[:,0:shape_x,1] = img_new[:,0:shape_x,2] = img_new[:,shape_x:2*shape_x,0] = img_new[:,shape_x:2*shape_x,1] = img_new[:,shape_x:2*shape_x,2] = img.getRaw()
            scipy.misc.toimage(img_new, cmin=0.0, cmax=1.0).save(config.outputFolder + config.diagnosis[0] + '\\images\\' + args.img_prefix + str(t) + '.jpg')
            tifffile.imsave(config.outputFolder + config.diagnosis[0] + '\\masks\\' + args.mask_prefix + str(t) + '.tif',img.getMask(),dtype=np.uint8)
    e=1

main()