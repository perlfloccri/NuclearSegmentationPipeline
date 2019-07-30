from matplotlib import pyplot as plt
import os
import argparse
import glob
import numpy as np
import pickle
import csv
from tifffile import imread
import sys
sys.path.append(r'D:\DeepLearning\DataGenerator')
from Classes.Image import Image
from Classes.Helper import Tools
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Tools.Helper import calculateAggregatedJaccardIndex, objectBasedMeasures, objectBasedMeasures4, jaccardIndex_with_object, jaccardIndex_with_area, getMetrics
from Tools.StructuredEvaluation import StructuredEvaluation,TestImage,Architecture,Metric,Diagnosis, Preparation, ChallengeLevel
from skimage.measure import label

def main():
    tools = Tools()
    structuredEvaluation = StructuredEvaluation()
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--inputFolder', help='Select input folder.', default=None)
    parser.add_argument('--outputFolder', help='select output folder', default=None)
    parser.add_argument('--resultfile',help="select resultfile",default=None)
    args = parser.parse_args()

    scaled_prediction_parameters = []
    notscaled_prediction_parameters = []
    scaled_predictions = []
    notscaled_predictions = []

    # Read all predictions
    ids_predictions = glob.glob(os.path.join(args.inputFolder,'*_reconstructed.pkl'))
    # Read groundtruth
    path_to_img = []
    tiles = []
    groundtruth_images = []
    groundtruth_masks = []

    with open(args.resultfile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            path_to_img.append(row[0])
            tiles.append(int(row[2]))
    for i in range (0, tiles.__len__()):
        if (tiles[i] == 0):
            # Evaluate also silver trained images against gold
            groundtruth_images.append(Image.pre_process_img(imread(path_to_img[i]), color='gray'))
            groundtruth_masks.append(Image.pre_process_img(imread(path_to_img[i].replace('\\images','\\masks')), color='gray'))

	# Annotations  used for the TMI paper - tags for each image, based on the image position
    diag = [Diagnosis.HaCaT,Diagnosis.HaCaT,Diagnosis.HaCaT,Diagnosis.HaCaT,Diagnosis.HaCaT,Diagnosis.HaCaT,Diagnosis.Neuroblastoma,Diagnosis.Neuroblastoma,Diagnosis.Neuroblastoma,Diagnosis.Neuroblastoma,Diagnosis.Ganglioneuroma]
    prep = [Preparation.cellline_cytospin,Preparation.cellline_cytospin,Preparation.cellline_grown,Preparation.cellline_cytospin,Preparation.cellline_cytospin,Preparation.cellline_cytospin,Preparation.tumor_touchimprint,Preparation.tumor_touchimprint,Preparation.tumor_touchimprint,Preparation.bonemarrow_cytospin,Preparation.tissuesection]
    level = [ChallengeLevel.low,ChallengeLevel.low,ChallengeLevel.low,ChallengeLevel.low,ChallengeLevel.medium,ChallengeLevel.low,ChallengeLevel.low,ChallengeLevel.low,ChallengeLevel.low,ChallengeLevel.medium,ChallengeLevel.high]


    for i in range(0,diag.__len__()):
        structuredEvaluation.addTestImage(TestImage(position = i, diagnosis = diag[i], preparation = prep[i], level = level[i]))

    for i in ids_predictions:
        if "_notscaled" in i:
            #notscaled_prediction_parameters.append(os.path.basename(i).split('_predictions')[0])
            #notscaled_predictions.append(pickle.load(open(i,"rb")))
            scaled_prediction_parameters.append(os.path.basename(i).split('_predictions')[0])
            scaled_predictions.append(pickle.load(open(i, "rb")))
        elif "_scaled" in i:
            scaled_prediction_parameters.append(os.path.basename(i).split('_predictions')[0])
            scaled_predictions.append(pickle.load(open(i, "rb")))

    # Postprocessing
    threshold_objectsize = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,1000,50]  # Threshold depending on magnification
    for index,elem in enumerate(scaled_predictions):
        for j in range(0,threshold_objectsize.__len__()):
            scaled_predictions[index]["masks"][j] = tools.postprocess_mask(label(scaled_predictions[index]["masks"][j]),threshold=threshold_objectsize[j])>0
        structuredEvaluation.addArchitecture(Architecture(name = scaled_prediction_parameters[index]))
    vals = np.linspace(0.3, 0.8, 256)
    np.random.shuffle(vals)
    vals[0] = 0
    cmap = plt.cm.colors.ListedColormap(plt.cm.CMRmap(vals))

    target="masks"
    detections = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    metrics = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for img_nr in [0,1,2,3,4,5,6,7,8,9,10]:
        #img_nr=0
        fig = plt.figure(img_nr+1);

        fig.suptitle("Scaled predictions", fontsize=16)
        cnt = 0
        for index,elem in enumerate(scaled_predictions):
            ax=plt.subplot(int(round(np.sqrt(scaled_predictions.__len__())))+1,int(round(np.sqrt(scaled_predictions.__len__())))+1,index+1)
            ax.imshow(label(scaled_predictions[index][target][img_nr]),cmap=cmap)
            ax.imshow(scaled_predictions[index][target][img_nr])
            ax.set_title(scaled_prediction_parameters[index], fontsize=5)
            cnt = cnt + 1
            erg = objectBasedMeasures4(groundtruth_masks[img_nr] * 255, label(scaled_predictions[index][target][img_nr]))
            [AJI_C,AJI_U] = calculateAggregatedJaccardIndex(groundtruth_masks[img_nr] * 255, label(scaled_predictions[index][target][img_nr]))
            detections[index].append(erg)
            metrics[index].append(getMetrics(erg["masks"]))
            results = getMetrics(erg["masks"])
            structuredEvaluation.addMetric(Metric(FP=results["FP"],TP=results["TP"],FN=results["FN"], dice = erg["DICE"], ji = erg["JI"],AJI_C = AJI_C, AJI_U = AJI_U,US = results["US"], OS = results["OS"]),image=img_nr,architecture=scaled_prediction_parameters[index])
        ax=plt.subplot(int(round(np.sqrt(scaled_predictions.__len__())))+1,int(round(np.sqrt(scaled_predictions.__len__())))+1,cnt+1)
        ax.imshow(groundtruth_masks[img_nr],cmap=cmap)
        ax.set_title('Groundtruth', fontsize=5)
        plt.show(block=False)
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__HaCaT_300epochs.csv", structuredEvaluation.calculateMetricsForDiagnosis(target='diagnosis', targetlist=[Diagnosis.HaCaT]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__NB_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='diagnosis', targetlist=[Diagnosis.Neuroblastoma]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__Alldiagnosis_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='diagnosis', targetlist=[Diagnosis.HaCaT, Diagnosis.Neuroblastoma, Diagnosis.Ganglioneuroma]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__GN_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='diagnosis', targetlist=[Diagnosis.Ganglioneuroma]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__cellline_cytospin_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='preparation', targetlist=[Preparation.cellline_cytospin]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__tumortouch_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='preparation', targetlist=[Preparation.tumor_touchimprint]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__cellline_grown_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='preparation', targetlist=[Preparation.cellline_grown]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__bonemarrow_cytospin_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='preparation', targetlist=[Preparation.bonemarrow_cytospin]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__lowlevel_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='level', targetlist=[ChallengeLevel.low]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__midlevel_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='level', targetlist=[ChallengeLevel.medium]))
    structuredEvaluation.printMetrics(r"D:\DeepLearning\Results\finalresults_allnetworks_gold__highlevel_300epochs.csv",structuredEvaluation.calculateMetricsForDiagnosis(target='level', targetlist=[ChallengeLevel.high]))
    pickle.dump(detections, open(r"D:\DeepLearning\Results\Results_gold\finalresults_allnetworks_gold_evaluation_masks.pkl", "wb"))

if __name__ == '__main__':
    main()
