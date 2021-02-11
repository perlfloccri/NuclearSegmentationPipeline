from enum import Enum
import numpy as np
import csv

class StructuredEvaluation:
    # Add and output structured evaluations for segmentation problems on multiple frameworks
    # written by Florian Kromp
    # 08.02.2019

    architectures = []
    images = []
    metrics = []

    def addArchitecture(self,architecture):
        self.architectures.append(architecture)

    def addTestImage(self,image):
        self.images.append(image)
        e=1

    def getArchitectureID(self,name):
        id = -1
        for index,elem in enumerate(self.architectures):
            if elem.getName() == name:
                id=index
        return id

    def getImageID(self, id):
        idx = -1
        for index,elem in enumerate(self.images):
            if elem.getPosition() == id:
                idx=index
        return idx

    def addMetric(self,metric,image=0,architecture=0):
        metric.setImageID(self.getImageID(image))
        metric.setArchitectureID(self.getArchitectureID(architecture))
        self.metrics.append(metric)

    def getAllMetricsForImage(self,image):
        metrics = []
        for index,elem in enumerate(self.metrics):
            if elem.getImageID() == image:
                metrics.append(elem)
        return metrics

    def calculateMetricsForDiagnosis(self,target,targetlist):
        image_ids = []
        metrics = []
        if target == 'diagnosis':
            for index,elem in enumerate(self.images):
                if (elem.getDiagnosis() in targetlist):
                    image_ids.append(index)
        elif target == 'preparation':
            for index,elem in enumerate(self.images):
                if (elem.getPreparation() in targetlist):
                    image_ids.append(index)
        elif target == 'level':
            for index,elem in enumerate(self.images):
                if (elem.getLevel() in targetlist):
                    image_ids.append(index)
        elif target == 'naturepaperclass':
            for index,elem in enumerate(self.images):
                if (NaturepaperClasses[elem.getNaturepaperClass()] in targetlist):
                    image_ids.append(index)
        elif target == 'magnification':
            for index,elem in enumerate(self.images):
                if (elem.getMagnification() in targetlist):
                    image_ids.append(index)
        elif target == 'modality':
            for index,elem in enumerate(self.images):
                if (elem.getModality() in targetlist):
                    image_ids.append(index)
        elif target == 'signal_to_noise':
            for index,elem in enumerate(self.images):
                if (elem.getSignalToNoise() in targetlist):
                    image_ids.append(index)
        elif target == 'challengelevel':
            for index, elem in enumerate(self.images):
                if (elem.getChallengeLevel() in targetlist):
                    image_ids.append(index)
        elif target == 'naturepaperclasstmi':
            for index,elem in enumerate(self.images):
                if (NaturepaperClasses[elem.getNaturepaperClass()] in targetlist):
                    image_ids.append(index)
        for index_arch,elem_arch in enumerate(self.architectures):
            name = elem_arch.getName()
            TP = 0
            FP = 0
            FN = 0
            US = 0
            OS = 0
            DICE = []
            JI = []
            AJI_C = 0
            AJI_U = 0

            for index_metr,elem_metr in enumerate(self.metrics):
                if ((elem_metr.getImageID() in image_ids) and (self.getArchitectureID(elem_arch.getName()) == elem_metr.getArchitectureID())):
                    TP += elem_metr.TP
                    FP += elem_metr.FP
                    FN += elem_metr.FN
                    US += elem_metr.US
                    OS += elem_metr.OS
                    DICE.extend(elem_metr.dice)
                    JI.extend(elem_metr.ji)
                    AJI_C += elem_metr.AJI_C
                    AJI_U += elem_metr.AJI_U
            metrics.append(finalMetricForPrint(FP=FP,TP=TP,FN=FN, dice = DICE, ji = JI,AJI_C = AJI_C, AJI_U = AJI_U, US = US, OS = OS, name = name, target = target, targetlist = targetlist, image_ids = image_ids))
        return metrics

    def calculateMetricsForDiagnosisCombined(self,target,targetlist, target2, targetlist2):
        image_ids = []
        image_ids2 = []
        metrics = []
        if target == 'diagnosis':
            for index,elem in enumerate(self.images):
                if (elem.getDiagnosis() in targetlist):
                    image_ids.append(index)
        elif target == 'preparation':
            for index,elem in enumerate(self.images):
                if (elem.getPreparation() in targetlist):
                    image_ids.append(index)
        elif target == 'level':
            for index,elem in enumerate(self.images):
                if (elem.getLevel() in targetlist):
                    image_ids.append(index)
        elif target == 'naturepaperclass':
            for index,elem in enumerate(self.images):
                if (NaturepaperClasses[elem.getNaturepaperClass()] in targetlist):
                    image_ids.append(index)
        elif target == 'magnification':
            for index,elem in enumerate(self.images):
                if (elem.getMagnification() in targetlist):
                    image_ids.append(index)
        elif target == 'modality':
            for index,elem in enumerate(self.images):
                if (elem.getModality() in targetlist):
                    image_ids.append(index)
        elif target == 'signal_to_noise':
            for index,elem in enumerate(self.images):
                if (elem.getSignalToNoise() in targetlist):
                    image_ids.append(index)
        elif target == 'challengelevel':
            for index, elem in enumerate(self.images):
                if (elem.getChallengeLevel() in targetlist):
                    image_ids.append(index)
        elif target == 'naturepaperclasstmi':
            for index,elem in enumerate(self.images):
                if (NaturepaperClasses[elem.getNaturepaperClass()] in targetlist):
                    image_ids.append(index)

        if target2 == 'diagnosis':
            for index, elem in enumerate(self.images):
                if (elem.getDiagnosis() in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'preparation':
            for index, elem in enumerate(self.images):
                if (elem.getPreparation() in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'level':
            for index, elem in enumerate(self.images):
                if (elem.getLevel() in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'naturepaperclass':
            for index, elem in enumerate(self.images):
                if (NaturepaperClasses[elem.getNaturepaperClass()] in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'magnification':
            for index, elem in enumerate(self.images):
                if (elem.getMagnification() in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'modality':
            for index, elem in enumerate(self.images):
                if (elem.getModality() in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'signal_to_noise':
            for index, elem in enumerate(self.images):
                if (elem.getSignalToNoise() in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'challengelevel':
            for index, elem in enumerate(self.images):
                if (elem.getChallengeLevel() in targetlist2):
                    image_ids2.append(index)
        elif target2 == 'naturepaperclasstmi':
            for index, elem in enumerate(self.images):
                if (NaturepaperClasses[elem.getNaturepaperClass()] in targetlist2):
                    image_ids2.append(index)
        image_ids = list(set(image_ids).intersection(set(image_ids2)))
        for index_arch,elem_arch in enumerate(self.architectures):
            name = elem_arch.getName()
            TP = 0
            FP = 0
            FN = 0
            US = 0
            OS = 0
            DICE = []
            JI = []
            AJI_C = 0
            AJI_U = 0

            for index_metr,elem_metr in enumerate(self.metrics):
                if ((elem_metr.getImageID() in image_ids) and (self.getArchitectureID(elem_arch.getName()) == elem_metr.getArchitectureID())):
                    TP += elem_metr.TP
                    FP += elem_metr.FP
                    FN += elem_metr.FN
                    US += elem_metr.US
                    OS += elem_metr.OS
                    DICE.extend(elem_metr.dice)
                    JI.extend(elem_metr.ji)
                    AJI_C += elem_metr.AJI_C
                    AJI_U += elem_metr.AJI_U
            metrics.append(finalMetricForPrint(FP=FP,TP=TP,FN=FN, dice = DICE, ji = JI,AJI_C = AJI_C, AJI_U = AJI_U, US = US, OS = OS, name = name, target = target, targetlist = targetlist, image_ids = image_ids))
        return metrics

    def printMetrics(self,filepath,metrics):
        with open(filepath, mode='w') as csv_file:
            fieldnames = ['architecture', 'target', 'targetlist','image_ids','FP', 'TP', 'FN', 'US_RATE', 'OS_RATE', 'OL_RECALL', 'OL_PRECISION', 'OL_F1','OL_MEAN_DICE','OL_MEAN_JI', 'AJI']
            writer = csv.DictWriter(csv_file, delimiter=';',fieldnames=fieldnames)
            writer.writeheader()
            target_string = ''
            image_id_string = ''
            for index,elem in enumerate(metrics[0].targetlist):
                if index == 0:
                    target_string += str(elem)
                else:
                    target_string += '_' + str(elem)
            for index,elem in enumerate(metrics[0].image_ids):
                if index == 0:
                    image_id_string += str(elem)
                else:
                    image_id_string += '_' + str(elem)
            for metric in metrics:
                writer.writerow({'architecture':metric.name, 'target':metric.target,'targetlist':target_string, 'image_ids':image_id_string,'FP':metric.FP, 'TP':metric.TP, 'FN':metric.FN, 'US_RATE':metric.US_RATE, 'OS_RATE':metric.OS_RATE, 'OL_RECALL':metric.RECALL, 'OL_PRECISION':metric.PRECISION, 'OL_F1':metric.F1,'OL_MEAN_DICE':metric.MEAN_DICE,'OL_MEAN_JI':metric.MEAN_JI, 'AJI':metric.AJI})


class TMIImage:
    # Class for adding annotated images and descriptions for segmentation evaluation paper revision
    # written by Florian Kromp
    # 29.04.2020
    position = 0
    diagnosis = ''
    preparation = ''
    magnification = ''
    modality = ''
    signal_to_noise = ''
    naturepaperclass = ''

    def __init__(self,position = 0,diagnosis = '', preparation = '', magnification = '', modality = '', signal_to_noise = '', naturepaperclass = '', challengelevel = ''):
        self.position = position
        self.diagnosis = diagnosis
        self.preparation = preparation
        self.magnification = magnification
        self.modality = modality
        self.signal_to_noise = signal_to_noise
        self.naturepaperclass = naturepaperclass
        self.challengelevel = challengelevel

    def getPosition(self):
        return self.position

    def getDiagnosis(self):
        return self.diagnosis

    def getPreparation(self):
        return self.preparation

    def getMagnification(self):
        return self.magnification

    def getSignalToNoise(self):
        return self.signal_to_noise

    def getNaturepaperClass(self):
        return self.naturepaperclass

    def getModality(self):
        return self.modality

    def getChallengeLevel(self):
        return self.challengelevel

class TestImage:
    # Class for adding annotated images and descriptions for segmentation evaluation paper
    # written by Florian Kromp
    # 08.02.2019
    position = 0
    diagnosis = ''
    preparation = ''
    level = ''

    def __init__(self,position = 0,diagnosis = '', preparation = '', level = '', naturepaperclass = ''):
        self.position = position
        self.diagnosis = diagnosis
        self.preparation = preparation
        self.level = level
        self.naturepaperclass = naturepaperclass

    def getPosition(self):
        return self.position

    def getDiagnosis(self):
        return self.diagnosis

    def getPreparation(self):
        return self.preparation

    def getLevel(self):
        return self.level

    def getNaturepaperClass(self):
        return self.naturepaperclass

class Diagnosis(Enum):
    HaCaT = 1
    Neuroblastoma = 2
    Ganglioneuroma = 3

class Preparation(Enum):
    cellline_cytospin = 1
    tumor_touchimprint = 2
    cellline_grown = 3
    bonemarrow_cytospin = 4
    tissuesection = 5

class ChallengeLevel(Enum):
    low = 1
    medium = 2
    high = 3

class NaturepaperClasses(Enum):
    GNB_I = 1
    GNB_II = 2
    NB_I = 3
    NB_II = 4
    NB_III = 5
    NB_IV = 6
    NC_I = 7
    NC_II = 8
    NC_III = 9
    TS = 10

class NaturepaperClassesTotal(Enum):
    Ganglioneuroma = 1
    Ganglioneuroma_differentconditions = 2
    Neuroblastoma_bmcytospin = 3
    Neuroblastoma_cellline_differentconditions = 4
    Neuroblastoma_cellline_LSM = 5
    Neuroblastoma_touchimprint = 6
    normal_cyto = 7
    normal_differentconditions = 8
    normal_grown = 9
    otherspecimen_tissuesections = 10
    Neuroblastoma = 11
    normal=12

class Architecture:
    # Class for adding annotated images and descriptions for segmentation evaluation paper
    # written by Florian Kromp
    # 08.02.2019
    name = []

    def __init__(self,name = ''):
        self.name = name

    def getName(self):
        return self.name


class Metric:
    # Class describing metrics for an image and the image and architecture they refer to
    # written by Florian Kromp
    # 08.02.2019
    FP = 0
    TP = 0
    FN = 0
    US = 0
    OS = 0
    dice = []
    ji = []
    AJI_C = 0
    AJI_U = 0
    image_id = 0
    architecture_id = 0

    def __init__(self,FP=0,TP=0,FN=0, dice = [], ji = [],AJI_C = 0, AJI_U = 0, US = 0, OS = 0, image_id=0,architecture_id=0):
        self.FP = FP
        self.TP = TP
        self.FN = FN
        self.US = US
        self.OS = OS
        self.dice = dice
        self.ji = ji
        self.AJI_C = AJI_C
        self.AJI_U = AJI_U
        self.image_id = image_id
        self.architecture_id = architecture_id

    def setImageID(self,id):
        self.image_id = id

    def setArchitectureID(self,id):
        self.architecture_id = id

    def getImageID(self):
        return self.image_id

    def getArchitectureID(self):
        return self.architecture_id


class finalMetricForPrint:
    # Class describing metrics for an image and the image and architecture they refer to
    # written by Florian Kromp
    # 08.02.2019
    FP = 0
    TP = 0
    FN = 0
    US_RATE = 0
    OS_RATE = 0
    RECALL = 0
    PRECISION = 0
    F1 = 0
    MEAN_DICE = 0
    MEAN_JI = 0
    AJI = 0
    name = 0
    target = ''
    targetlist = []
    image_ids = []

    def __init__(self,FP=0,TP=0,FN=0, dice = [], ji = [],AJI_C = 0, AJI_U = 0, US = 0, OS = 0, name = '', target = '', targetlist = [], image_ids = []):
        self.FP = FP
        self.TP = TP
        self.FN = FN
        self.MEAN_DICE = np.mean(dice)
        self.MEAN_JI = np.mean(ji)
        if AJI_U == 0:
            self.AJI = 0
        else:
            self.AJI = AJI_C / AJI_U
        if (TP+FN)>0:
            self.US_RATE = US / (TP+FN)
            self.OS_RATE = OS / (TP+FN)
            self.RECALL = TP / (TP+FN)
        else:
            self.US_RATE = 0
            self.OS_RATE = 0
            self.RECALL = 0
        self.image_ids = image_ids
        self.name = name
        self.target = target
        self.targetlist = targetlist
        if (TP + FP)>0:
            self.PRECISION = TP / (TP + FP)

        try:
            self.F1 = 2 * self.PRECISION * self.RECALL / (self.PRECISION + self.RECALL)
        except:
            self.F1 = 0