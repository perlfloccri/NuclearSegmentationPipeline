from Classes.DBTools import TisQuantExtract
from Classes.Config import Config
from Classes.Helper import Tools
from Classes.Image import AnnotatedImage,AnnotatedObjectSet, ArtificialAnnotatedImage
from matplotlib import pyplot as plt
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--tissue', help='select tissue to train.', default=None)
    parser.add_argument('--inputFolder', help='Select input folder.', default=None)
    parser.add_argument('--outputFolder', help='select output folder', default=None)
    parser.add_argument('--scale', help='select output folder', default=None)
    parser.add_argument('--mode', help='select output folder', default='train')
    parser.add_argument('--resultsfile', help='select output folder', default=None)
    parser.add_argument('--overlap', help='select output folder', default=None)
    args = parser.parse_args()
    tisquant = TisQuantExtract()
    config = Config
    if args.tissue:
        config.diagnosis = [args.tissue]

    if args.outputFolder:
        config.outputFolder = args.outputFolder
    if args.scale == '1':
        config.scale=True
    if args.mode:
        config.mode=args.mode
    if args.resultsfile:
        config.resultsfile=args.resultsfile
    if args.overlap:
        config.overlap=int(args.overlap)

    print(config.diagnosis)
    print(config.outputFolder)
    print(config.scale)
    tools = Tools()

    annotated_nuclei = AnnotatedObjectSet()
    ids_images = glob.glob(os.path.join(args.inputFolder,config.diagnosis[0],'images','*.tif'))
    ids_masks = glob.glob(os.path.join(args.inputFolder, config.diagnosis[0], 'masks', '*.tif'))

    # Create dataset for training the pix2pix-network based on image pairs
    #for index,elem in enumerate(ids_paths):
    for index, elem in enumerate(ids_images):
        test = AnnotatedImage()
        test.readFromPath(ids_images[index], ids_masks[index],type='uint16')
        #enhanced_images = tools.enhanceImage(test,flip_left_right=True,flip_up_down=True,deform=True)
        #for index,img in enumerate(enhanced_images):
        #    annotated_nuclei.addObjectImage(img,useBorderObjects=config.useBorderObjects)
        annotated_nuclei.addObjectImage(test, useBorderObjects=config.useBorderObjects,path_to_img=ids_images[index])
    # Create and save the image tiles
    tools.createAndSaveTiles(annotated_nuclei,config)

main()