from matplotlib import pyplot as plt
import os
import argparse
import glob
import numpy as np
import pickle
import csv
from tifffile import tifffile
import sys
sys.path.append(r'E:\NuclearSegmentationPipeline\DataGenerator')
from Classes.Image import Image
from Classes.Helper import Tools
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Tools.Helper import calculateAggregatedJaccardIndex, objectBasedMeasures, objectBasedMeasures4, jaccardIndex_with_object, jaccardIndex_with_area, getMetrics, getminObjectSize
from Tools.StructuredEvaluation import StructuredEvaluation,TMIImage,Architecture,Metric,Diagnosis, Preparation, ChallengeLevel, NaturepaperClasses
from skimage.measure import label
import matplotlib
import re

def main():
    tools = Tools()
    structuredEvaluation = StructuredEvaluation()
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--resultfile', help='select result file', default=r"E:\NuclearSegmentationPipeline\Results\results_scaled.csv")
    args = parser.parse_args()

    vals = np.linspace(0.1, 0.9, 256)
    np.random.shuffle(vals)
    vals[0] = 1
    cmap = plt.cm.colors.ListedColormap(plt.cm.cubehelix(vals))

    # read csv file and images
    raw_images = []
    groundtruth = []
    basefolder = r"D:\DeepLearning\Results_revision\Dataset_revision"

    reader = csv.reader(open(r"E:\NuclearSegmentationPipeline\DataGenerator\image_description_final_revision.csv", 'r'))
    next(reader)
    abs_index = 0
    mapping = dict()
    try:
        for row in reader:
            entrys = row[0].split(';')
            #if entrys[3] == 'test':
            mapping[entrys[0]] = row[0]
    except:
        print('Unable to open csv file')

    # Update evaluation according to image position
    type_list = ["Ganglioneuroblastoma", "Ganglioneuroblastoma_differentconditions",
                 "Neuroblastoma_bmcytospin", "Neuroblastoma_cellline_differentconditions", "Neuroblastoma_cellline_LSM",
                 "Neuroblastoma_touchimprint", "normal_cyto", "normal_differentconditions", "normal_grown",
                 "otherspecimen_tissuesections"]
    base_path = r"D:\DeepLearning\DataGenerator\tisquant_train_val_test_gold_revision\test"
    abs_index = 0

    path_to_img = []
    tiles = []
    scales = []
    count = 0
    # Read images from tiling file
    with open(args.resultfile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if int(row[2]) == 0 and count <37:
                raw_images.append(tifffile.imread(row[0]))
                groundtruth.append(tifffile.imread(row[0].replace('images', 'masks')))
                scales.append(row[1])
                entrys = mapping[os.path.basename(row[0]).split('.')[0]].split(';')
                structuredEvaluation.addTestImage(
                    TMIImage(position=abs_index, diagnosis=entrys[1], preparation=entrys[2], magnification=entrys[5],
                             modality=entrys[7], signal_to_noise=entrys[12], naturepaperclass=entrys[4], challengelevel=entrys[14]))
                abs_index = abs_index + 1

    min_objects = []
    min_label = []
    for img_nr in range(0,abs_index):
        minsize,labeli=getminObjectSize(groundtruth[img_nr])
        min_objects.append(minsize)
        min_label.append(labeli)

    target = "masks"
    #ids_predictions = glob.glob(os.path.join(r"D:\DeepLearning\Results_revision\Results_gold", '*_reconstructed.pkl'))
    ids_predictions = glob.glob(os.path.join(r"E:\NuclearSegmentationPipeline\Results\\", '*.pkl'))

    predictions = []
    prediction_parameters = []

    for i in ids_predictions:
        prediction_parameters.append(os.path.basename(i).split('_reconstructed')[0])
        #predictions.append(pickle.load(open(i, "rb")))
        # Workaround for rwf pickles
        with open(i, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            predictions.append(u.load())

    for index,elem in enumerate(predictions):
        for j in range(0,min_objects.__len__()):
            predictions[index]["masks"][j] = tools.postprocess_mask(label(predictions[index]["masks"][j]),threshold=min_objects[j])
        structuredEvaluation.addArchitecture(Architecture(name = prediction_parameters[index]))

    for img_nr in range(0,abs_index):
        print("Calcualting metrics for image " + str(img_nr) + "/" + str(abs_index))
        cnt = 0
        for index,elem in enumerate(predictions):

            erg = objectBasedMeasures4(groundtruth[img_nr] * 255, predictions[index][target][img_nr])
            [AJI_C,AJI_U] = calculateAggregatedJaccardIndex(groundtruth[img_nr] * 255, predictions[index][target][img_nr])
            results = getMetrics(erg["masks"])
            structuredEvaluation.addMetric(Metric(FP=results["FP"],TP=results["TP"],FN=results["FN"], dice = erg["DICE"], ji = erg["JI"],AJI_C = AJI_C, AJI_U = AJI_U,US = results["US"], OS = results["OS"]),image=img_nr,architecture=prediction_parameters[index])

    structuredEvaluation.printMetrics(r"E:\NuclearSegmentationPipeline\Results\test_GNB.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='naturepaperclass', targetlist=[NaturepaperClasses.GNB_I]))
    structuredEvaluation.printMetrics(r"E:\NuclearSegmentationPipeline\Results\test_NB.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='naturepaperclass', targetlist=[NaturepaperClasses.NB_I,NaturepaperClasses.NB_IV]))
    structuredEvaluation.printMetrics(r"E:\NuclearSegmentationPipeline\Results\test_normal.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='naturepaperclass', targetlist=[NaturepaperClasses.NC_I,NaturepaperClasses.NC_III]))




    e=1
main()