from matplotlib import pyplot as plt
import os
import argparse
import glob
import numpy as np
import pickle
import csv
from tifffile import imread
import sys
sys.path.append(r'E:\NuclearSegmentationPipeline\DataGenerator')
from Classes.Image import Image
from Classes.Helper import Tools
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Tools.Helper import calculateAggregatedJaccardIndex, objectBasedMeasures, objectBasedMeasures4, jaccardIndex_with_object, jaccardIndex_with_area, getMetrics, calculateSinglecellDiceJI
from Tools.StructuredEvaluation import StructuredEvaluation,TestImage,Architecture,Metric,Diagnosis, Preparation, ChallengeLevel, NaturepaperClasses
from skimage.measure import label

def main():
    tools = Tools()
    structuredEvaluation = StructuredEvaluation()
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--inputFolder', help='Select input folder.', default=None)
    parser.add_argument('--outputFolder', help='select output folder', default=None)
    parser.add_argument('--resultfile', help='select result file', default=r"E:\NuclearSegmentationPipeline\Results\results_scaled.csv")
    args = parser.parse_args()

    # Read csv file to get position
    mapping_images = dict()
    natureclass = dict()
    abs_index = 0
    mapping_class = dict()

    with open(args.resultfile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if int(row[2]) == 0:
                mapping_images[os.path.basename(row[0])] = abs_index
                abs_index = abs_index + 1
    # Read csv file to get naturepaperclass
    with open(r"E:\NuclearSegmentationPipeline\DataGenerator\image_description_final_revision.csv") as csv_file2:
        csv_reader2 = csv.reader(csv_file2)
        for row in csv_reader2:
            mapping_class[row[0].split(';')[0]] = row[0].split(';')[1]
            natureclass[row[0].split(';')[0]] = row[0].split(';')[4]
    rawimages = []
    groundtruth = []
    img_pathes = []
    base_path = r"E:\NuclearSegmentationPipeline\DataGenerator\dataset_singlecellgroundtruth"

    type_list = ["Ganglioneuroma", "Neuroblastoma", "normal"]

    abs_index = 0
    target = "masks"

    ids_predictions = glob.glob(os.path.join(r"E:\NuclearSegmentationPipeline\Results\\", '*.pkl'))
    predictions = []
    prediction_parameters = []
    print("Loading predictions...")
    for i in ids_predictions:
        prediction_parameters.append(os.path.basename(i).split('_reconstructed')[0])
        #predictions.append(pickle.load(open(i, "rb")))
        # Workaround for rwf pickles
        with open(i, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            predictions.append(u.load())

    for index,elem in enumerate(predictions):
        structuredEvaluation.addArchitecture(Architecture(name = prediction_parameters[index]))
    type_list_sorted = []
    img_name_list = []
    abs_index = 0
    print ("Reading images ...")
    for type in type_list:
        imgs = glob.glob(os.path.join(base_path, type, "images", "*.tif"))
        for index,img in enumerate(imgs):
            groundtruth.append(imread(img.replace('images','masks').replace('.tif','singlemask.tif')))
            img_name_list.append(os.path.basename(img))
            type_tmp = type
            structuredEvaluation.addTestImage(TestImage(position = abs_index, naturepaperclass=natureclass[os.path.basename(img).replace('.tif','')]))
            abs_index = abs_index + 1
    print("Start prediction")
    for index in range(0, groundtruth.__len__()):
        print("Calculating metrics for image number " + str(index) + " from " + str(groundtruth.__len__()) + " ...")
        for ix,prediction in enumerate(predictions):
            # Calculate metrics for Evi
            pred_index = mapping_images[img_name_list[index]]
            erg = calculateSinglecellDiceJI(groundtruth[index],label(prediction['masks'][pred_index].astype(np.float32)))
            structuredEvaluation.addMetric(Metric(dice=erg["DICE"], ji=erg["JI"]), image=index, architecture=prediction_parameters[ix])

    structuredEvaluation.printMetrics(r"E:\NuclearSegmentationPipeline\Results\test_singlecellannotation_GNB.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='naturepaperclass', targetlist=[NaturepaperClasses.GNB_I]))
    structuredEvaluation.printMetrics(r"E:\NuclearSegmentationPipeline\Results\test_singlecellannotation__NB.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='naturepaperclass', targetlist=[NaturepaperClasses.NB_IV]))
    structuredEvaluation.printMetrics(r"E:\NuclearSegmentationPipeline\Results\test_singlecellannotation__normal.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='naturepaperclass', targetlist=[NaturepaperClasses.NC_I]))
    structuredEvaluation.printMetrics(r"E:\NuclearSegmentationPipeline\Results\test_singlecellannotation__alldiagnosis.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='naturepaperclass', targetlist=[NaturepaperClasses.GNB_I, NaturepaperClasses.NB_IV, NaturepaperClasses.NC_I]))

if __name__ == '__main__':
    main()