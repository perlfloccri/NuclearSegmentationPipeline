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
    args = parser.parse_args()
    tisquant = TisQuantExtract()
    config = Config
    if args.tissue:
        config.diagnosis = [args.tissue]

    if args.outputFolder:
        config.outputFolder = args.outputFolder

    if args.scale:
        config.scale=int(args.scale)

    if args.tissue == 'Ganglioneuroma':
        n_freq = 20
    else:
        n_freq = 30
    print("Scale: " + args.scale)

    print(config.diagnosis)
    print(config.outputFolder)
    tools = Tools()

    annotated_nuclei = AnnotatedObjectSet()
    #ids_paths = tisquant.dbconnector.execute(query=tisquant.getLevel3AnnotatedImagesByDiagnosis_Query(diagnosis = config.diagnosis,magnification = config.magnification, staining_type = config.staining_type, staining = config.staining, segmentation_function = config.segmentation_function, annotator = config.annotator, device = config.device))
    ids_images = glob.glob(os.path.join(args.inputFolder,config.diagnosis[0],'images','*.tif'))
    ids_masks = glob.glob(os.path.join(args.inputFolder, config.diagnosis[0], 'masks', '*.tif'))

    # Create dataset for training the pix2pix-network based on image pairs
    #for index,elem in enumerate(ids_paths):
    for index, elem in enumerate(ids_images):
        #groundtruth_paths = tisquant.dbconnector.execute(tisquant.getLevel3AnnotationByImageId_Query(elem[0],config.annotator)) # Pathes from groundtruth from all annotators
        #groundtruth_paths = tisquant.dbconnector.execute(tisquant.getLevel3AnnotationByImageIdUsingMaxExperience_Query(elem[0], config.annotator)) # Pathes from groundtruth from most experienced annotator
        #for groundtruth_path in groundtruth_paths:
        test = AnnotatedImage()
        #test.readFromPath(tools.getLocalDataPath(elem[1],1),tools.getLocalDataPath(groundtruth_path[0],3))
        test.readFromPath(ids_images[index], ids_masks[index],type='uint16')
        enhanced_images = tools.enhanceImage(test,flip_left_right=True,flip_up_down=True,deform=False)
        for index,img in enumerate(enhanced_images):
            annotated_nuclei.addObjectImage(img,useBorderObjects=config.useBorderObjects)

    # Create the image pairs
    tools.createPix2pixDataset(annotated_nuclei,config,n_freq = n_freq,tissue=args.tissue)

    e=1

main()