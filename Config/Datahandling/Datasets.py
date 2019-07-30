import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(r'D:\DeepLearning\Kaggle\Datahandling')
import utils_for_datasets
import glob
import numpy as np
import cv2
import re
import os.path
import scipy
import time
from skimage.measure import label
import skimage.transform as ski_transform
import matplotlib.pyplot as plt
DATASETROOT = 'CVSP\Cameratrap'
DATASETROOT_CVL = 'CVSP\CVL'
UNETROOT = 'D:\DeepLearning\Semantic_segmentation\Cameratrap_Dataset'
UNETROOT_CVL = 'D:\DeepLearning\Semantic_segmentation\CVL_Dataset'
DATASET_FOLDER_TISQUANT = r'D:\DeepLearning\SCCHCode\TisQuantValidation\data'
#DATASET_FOLDER_KAGGLE = r'D:\\DeepLearning\\SCCHCode\\data\\kaggle-dsbowl-2018-dataset-fixes-master\\stage1_train'
#DATASET_FOLDER_KAGGLE = r'D:\\DeepLearning\\SCCHCode\\data\\Kaggle\\stage1_train'
from scipy.io import loadmat, savemat
from tifffile import tifffile
from Config.Config import UNETSettings
from tqdm import tqdm

class TisquantDataset(utils_for_datasets.Dataset):

    def load_data(self,width=None,height=None,ids=None,mode=1):
        self.add_class("Nuclei",1,'Nucleus')
        if (mode==1):
            data_file = "256x256_TisQuantTrainingData_Evaluation1_new.mat"
        else:
            data_file = "256x256_TisQuantTestData_Evaluation1_new.mat"
        print('... LOADING DATA')
        Images, Labels, FileNames = [], [], []
        raw_data = loadmat(os.path.join(DATASET_FOLDER_TISQUANT, data_file), struct_as_record=True)
        if (mode==1):
            raw_data = raw_data['trainingset']
        else:
            raw_data = raw_data['testset']

        Images, Masks = [], []
        slice_size = 256
        masks = raw_data['groundtruth'][0]
        raw_images = raw_data['rawimage'][0]
        n_images = len(raw_images)
        for i,img in enumerate(raw_images):
            #img_new = np.zeros((3, img.shape[0], img.shape[1]))
            #img_new[0] = img
            #img_new[1] = img
            #img_new[2] = img
            #Images.append(img_new / 255.0)
            #Images.append(img / 255.0)
            #Images.append(img / 255.0)
            Images.append(img)
            #Masks.append(label(masks[i]>0))
            Masks.append(masks[i])
        # convert to conv net format
        img_size = Images[0].shape
        #Images = np.asarray(Images, dtype=np.float32).reshape(-1, img_size[0], img_size[1],img_size[2])
        #Images = np.transpose(Images, (0, 2, 3, 1))
        Images = np.asarray(Images, dtype=np.float32).reshape(-1, img_size[0], img_size[1])

        #Masks = np.asarray(Masks, dtype=np.float32).reshape(-1, 1, img_size[1], img_size[2])
        #Masks = np.transpose(Masks, (0, 2, 3, 1))
        Masks = np.asarray(Masks, dtype=np.float32).reshape(-1, img_size[0], img_size[1])


        train_val = 0.8
        ret_val = 0
        n_tr = int(round(Images.shape[0] * 0.8))
        ids = np.arange(Images.__len__())
        if (mode == 1): # Trainingset
            np.random.shuffle(ids)
        self.images = Images
        self.masks = Masks

        for i in range(self.images.shape[0]):
            self.add_image("Nuclei", image_id=i, path=None,width=width, height=height)
        self.train_cnt = int(self.images.__len__()*0.8)
        #self.images = np.transpose(self.images,(0,3,1,2))
        #self.masks = np.transpose(self.masks,(0,3,1,2))
        return ids

    def getMeanMaskObjectSize(self, image_id):
        masks = self.load_mask(image_id)
        masks_new = masks[0][:, :, 1:]
        print("Summe: {0}, Laenge: {1}".format(masks_new.sum(), masks_new.shape[2]))
        if (np.isnan(masks_new.sum() / masks_new.shape[2])):
            return 0
        else:
            return int(masks_new.sum() / masks_new.shape[2])

    def load_image(self, image_id):
        return self.images[image_id]

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = self.masks[image_id]
        count = int(mask.max())
        mask_new = np.zeros([info['height'], info['width'], count+1], dtype=np.uint8) # one more for background
        for i in range(count+1):
            #mask_new[:, :, i:i+1] = (mask == i).transpose(1, 2, 0)
            mask_new[:, :, i:i + 1] = (mask==i).reshape(mask.shape[0], mask.shape[1], -1)
     #       mask_new[:, :, i:i+1] = (mask==i).transpose(1,2,0)
        # Map class names to class IDs.
        class_ids = np.ones(count+1) # one more fore background

        #add Background
        #class_ids[count] = 0 # add Background
        #mask_new[:, :, count:count + 1] = (mask == 0).transpose(1, 2, 0)
        #class_ids[count] = 0  # add Background
        class_ids[0] = 0  # add Background
        # End add Background

        return mask_new, class_ids.astype(np.int32)

    def load_mask_one_layer(self,image_id):
        return self.masks[image_id]#[0]

class KaggleDataset(utils_for_datasets.Dataset):

    def load_data(self,width=None,height=None,ids=None,mode=1,folders=None):
        self.image_path = []
        self.mask_path = []
        self.add_class("Nucleus",1,'Nucleus')
        self.setImagePaths(folders)
        ids = np.arange(self.image_path.__len__())
        np.random.seed(1)
        np.random.shuffle(ids)
        self.ids = ids
        for i in self.ids:
            self.add_image("Nucleus", image_id=i, path=None)
        return ids

    def load_image(self, image_id):
        info = self.image_info[image_id]
        img = cv2.imread(self.image_path[self.ids[image_id]])
        #img = ski_transform.resize(img, (info['height'], info['width']), mode='reflect')
        return img

    def setImagePaths(self,folders=""):
        for folder in os.listdir(folders):
            file_pattern = os.path.join(folders,folder,'images',"*.png")
            #print(file_pattern)
            img_files = glob.glob(file_pattern)
            for i in img_files:
                self.image_path.append(i)
                self.mask_path.append(os.path.join(folders,folder,'masks'))

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        mask_path = self.mask_path[self.ids[image_id]]
        file_pattern = os.path.join(mask_path, "*.png")
        info = self.image_info[image_id]
        mask_files = glob.glob(file_pattern)
        #mask_tmp = cv2.imread(mask_files[0])
        mask_new = np.zeros([info['height'], info['width'], mask_files.__len__()+1], dtype=np.uint8) # one more for background
        count = 1
        mask_total = 0
        for i in mask_files:
            mask = cv2.imread(i)
            mask = mask[:, :, 1] / 255.0
            #mask = ski_transform.resize(mask, (info['height'], info['width']), mode='reflect')
            mask_new[:, :, count] = (mask)
            mask_total = mask_total + (mask>0) * count
            count = count + 1
        # Map class names to class IDs.
        class_ids = np.ones(count) # one more fore background
        #add Background
        class_ids[0] = 0;   # Background
        mask_new[:, :, 0] = np.invert(mask_total.astype(np.bool))
        # End add Background

        return mask_new, class_ids.astype(np.int32)

    def load_mask_one_layer(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        mask_path = self.mask_path[self.ids[image_id]]
        file_pattern = os.path.join(mask_path, "*.png")
        info = self.image_info[image_id]
        mask_files = glob.glob(file_pattern)
        #mask_tmp = cv2.imread(mask_files[0])
        mask_new = np.zeros([info['width'], info['height'], mask_files.__len__()+1], dtype=np.uint8) # one more for background
        count = 1
        mask_total = 0
        for i in mask_files:
            mask = cv2.imread(i)
            mask = mask[:, :, 1] / 255.0
            #mask = ski_transform.resize(mask, (info['height'], info['width']), mode='reflect')
            mask_new[:, :, count] = (mask)
            mask_total = mask_total * (mask == 0)
            mask_total = mask_total + (mask>0) * count
            count = count + 1
        return mask_total

    def getMeanMaskObjectSize(self, image_id):
        mask_path = self.mask_path[self.ids[image_id]]
        file_pattern = os.path.join(mask_path, "*.png")
        mask_files = glob.glob(file_pattern)
        total_sum = 0;
        for i in mask_files:
            mask = cv2.imread(i)
            total_sum = total_sum + (mask>0).sum()
        return (total_sum / mask_files.__len__()).astype(np.int16)

    def pre_process_img(self,img, color):
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

class ArtificialNucleiDataset(utils_for_datasets.Dataset):
    img_prefix = 'Img_'
    img_postfix = '-outputs.png'
    mask_prefix = 'Mask_'
    mask_postfix = '.tif'
    settings = UNETSettings()

    def load_data(self, width=256, height=256, ids=None, mode=1):
        # Load settings
        self.image_path = []
        self.mask_path = []

        self.add_class("ArtificialNuclei", 1, 'ArtificialNuclei')
        train_cnt = 0
        val_cnt = 0
        print("Loading train data ...")
        if self.settings.network_info["traintestmode"] == 'train':
            for i in self.settings.network_info["dataset_dirs_train"].split(';'):
                img_range = self.setImagePaths(folders=[i + "\\images"])
                self.setMaskPaths(folders=[i + "\\masks"],img_range=img_range)
            print("Checking train path ...")
            self.checkPath()
            print("Loading val data ...")
            train_cnt = self.image_path.__len__()
            for i in self.settings.network_info["dataset_dirs_val"].split(';'):
                img_range = self.setImagePaths(folders=[i + "\\images"])
                self.setMaskPaths(folders=[i + "\\masks"],img_range=img_range)
            print("Checking val path ...")
            self.checkPath()
            val_cnt += self.image_path.__len__() - train_cnt
            #ids = np.arange(self.image_path.__len__())
            ids_train = np.arange(0,train_cnt)
            ids_val = np.arange(train_cnt, train_cnt+val_cnt)
            self.train_cnt = train_cnt
            self.val_cnt = val_cnt
            np.random.shuffle(ids_train)
            np.random.shuffle(ids_val)
            self.ids = np.concatenate((ids_train,ids_val),axis=0)
        else:
            for i in self.settings.network_info["dataset_dirs_test"].split(';'):
                img_range = self.setImagePaths(folders=[i + "\\images"])
                self.setMaskPaths(folders=[i + "\\masks"],img_range=img_range)
            print("Checking train path ...")
            self.checkPath()
            self.ids = np.arange(0,self.image_path.__len__())
        for i in self.ids:
            self.add_image("ArtificialNuclei", image_id=i, path=None, width=width, height=height)
        return ids

    def checkPath(self):
        to_delete = []
        for index,i in tqdm(enumerate(self.image_path)):
            if not os.path.exists(i):
                to_delete.append(index)
        to_delete.sort(reverse=True)
        for i in to_delete:
            del self.image_path[i]
            del self.mask_path[i]

    def load_image(self, image_id):
        info = self.image_info[image_id]
        img_final = cv2.imread(self.image_path[self.ids[image_id]])
        try:
            img_final = img_final[:,:,0]
        except:
            None
        #return img_final / 255.0
        if self.settings.network_info["netinfo"] == 'maskrcnn': # mask rcnn need an rgb image
            img_new = np.zeros((img_final.shape[0],img_final.shape[1],3))
            img_new[:,:,0] = img_new[:,:,1] = img_new[:,:,2] = img_final
            img_final = img_new
        return img_final

    def setImagePaths(self, folders=""):
        for folder in folders:
            file_pattern = os.path.join(folder, self.img_prefix + "*" + self.img_postfix) #"Img_*-outputs.png")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            img_range = range(0,img_files.__len__())
            for i in img_range:
                #self.image_path.append(os.path.join(folder, "Img_" + str(i) + "-outputs.png"))
                self.image_path.append(os.path.join(folder, self.img_prefix + str(i) + self.img_postfix))
            # for i in img_files:
            #    self.image_path.append(i)
            return img_range

    def setMaskPaths(self, folders="",img_range=None):
        for folder in folders:
            file_pattern = os.path.join(folder, self.mask_prefix + "*" + self.mask_postfix) #"Mask_*.tif")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            #for i in range(0,img_files.__len__()):
            for i in img_range:
                self.mask_path.append(os.path.join(folder, self.mask_prefix + str(i) + self.mask_postfix))
                #self.mask_path.append(os.path.join(folder, "Mask_" + str(i) + ".tif"))


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = tifffile.imread(self.mask_path[self.ids[image_id]])

        if np.unique(mask).__len__() > 1:
            count = np.unique(mask).__len__()-1 # one less because of 0

            mask_new = np.zeros([info['height'], info['width'], count], dtype=np.uint8)  # one more for background
            running = 0
            for i in np.unique(mask): #range(1, count):
                if ((i > 0) & ((mask == i).sum() > 0)):
                    mask_new[:, :, running] = (mask == i)
                    running = running + 1
            # Map class names to class IDs.
            class_ids = np.ones(count)
        else:
            mask_new = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
            class_ids = np.zeros([1])
        return mask_new, class_ids.astype(np.int32)

    def load_mask_one_layer(self, image_id,relabel=False):
        mask = tifffile.imread(self.mask_path[self.ids[image_id]])
        if (mask.ndim > 2):
            mask = mask[:,:,0]
        if (relabel):
            mask_tmp = np.zeros((mask.shape[0],mask.shape[1]))
            running=1
            for i in np.unique(mask):
                if i > 0:
                    mask_tmp = mask_tmp + running * (mask==i)
                    running = running + 1
            mask = mask_tmp.astype(np.float)
        return mask #mask.astype(np.float)

    def pre_process_img(self, img, color):
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

    def split_train_test(self,width=256, height=256):
        dataset_train = ArtificialNucleiDataset()
        dataset_test = ArtificialNucleiDataset()
        dataset_train.image_path = []
        dataset_train.mask_path = []
        dataset_train.add_class("ArtificialNuclei", 1, 'ArtificialNuclei')
        dataset_test.image_path = []
        dataset_test.mask_path = []
        dataset_test.add_class("ArtificialNuclei", 1, 'ArtificialNuclei')
        image_path_train = []
        image_path_val = []
        mask_path_train = []
        mask_path_val = []
        self.ids = []
        run = 0
        dataset_train.image_path.extend(self.image_path[0:self.train_cnt])
        dataset_train.mask_path.extend(self.mask_path[0:self.train_cnt])
        dataset_train.train_cnt = self.image_path.__len__()

        dataset_test.image_path.extend(self.image_path[self.train_cnt:])
        dataset_test.mask_path.extend(self.mask_path[self.train_cnt:])
        dataset_test.train_cnt = self.image_path.__len__() - self.train_cnt

        ids_train = np.arange(0,self.train_cnt)
        ids_val = np.arange(0,self.val_cnt)
        np.random.shuffle(ids_train)
        np.random.shuffle(ids_val)
        dataset_train.ids = ids_train
        dataset_test.ids = ids_val

        for i in dataset_train.ids:
            dataset_train.add_image("ArtificialNuclei", image_id=i, path=None, width=width, height=height)
        for i in dataset_test.ids:
            dataset_test.add_image("ArtificialNuclei", image_id=i, path=None, width=width, height=height)

        dataset_train.prepare()
        dataset_test.prepare()
        return dataset_train, dataset_test

class TisquantDatasetNew(ArtificialNucleiDataset):

    def setImagePaths(self, folders=""):
        self.img_postfix = ".jpg"
        for folder in folders:
            #self.img_prefix = os.path.basename(folder) + "_"
            folder_names = folder.split('\\')
            self.img_prefix = "Img_" + folder_names[folder_names.__len__() - 2] + "_"
            file_pattern = os.path.join(folder, self.img_prefix + "*" + self.img_postfix) #"Img_*-outputs.png")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            img_range = range(0,img_files.__len__())
            for i in img_range:
                #self.image_path.append(os.path.join(folder, "Img_" + str(i) + "-outputs.png"))
                self.image_path.append(os.path.join(folder, self.img_prefix + str(i) + self.img_postfix))
            # for i in img_files:
            #    self.image_path.append(i)
            return img_range

    def setMaskPaths(self, folders="",img_range=None):
        self.mask_postfix = ".tif"
        for folder in folders:
            #self.mask_prefix = os.path.basename(folder) + "_"
            #self.mask_prefix = "Mask_"
            folder_names = folder.split('\\')
            self.mask_prefix = "Mask_" + folder_names[folder_names.__len__() - 2] + "_"
            file_pattern = os.path.join(folder, self.mask_prefix + "*" + self.mask_postfix) #"Mask_*.tif")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            #for i in range(0,img_files.__len__()):
            for i in img_range:
                self.mask_path.append(os.path.join(folder, self.mask_prefix + str(i) + self.mask_postfix))
                #self.mask_path.append(os.path.join(folder, "Mask_" + str(i) + ".tif"))

class SpecificNucleiDataset(ArtificialNucleiDataset):

    def setImagePaths(self, folders=""):
        self.img_postfix = "-outputs.png"
        for folder in folders:
            #self.img_prefix = os.path.basename(folder) + "_"
            folder_names = folder.split('\\')
            self.img_prefix = "Specific_"
            file_pattern = os.path.join(folder, self.img_prefix + "*" + self.img_postfix) #"Img_*-outputs.png")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            img_range = range(0,img_files.__len__())
            for i in img_range:
                #self.image_path.append(os.path.join(folder, "Img_" + str(i) + "-outputs.png"))
                self.image_path.append(os.path.join(folder, self.img_prefix + str(i) + self.img_postfix))
            # for i in img_files:
            #    self.image_path.append(i)
            return img_range

    def setMaskPaths(self, folders="",img_range=None):
        self.mask_postfix = ".tif"
        for folder in folders:
            #self.mask_prefix = os.path.basename(folder) + "_"
            #self.mask_prefix = "Mask_"
            folder_names = folder.split('\\')
            self.mask_prefix = "Specific_Mask_"
            file_pattern = os.path.join(folder, self.mask_prefix + "*" + self.mask_postfix) #"Mask_*.tif")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            #for i in range(0,img_files.__len__()):
            for i in img_range:
                self.mask_path.append(os.path.join(folder, self.mask_prefix + str(i) + self.mask_postfix))
                #self.mask_path.append(os.path.join(folder, "Mask_" + str(i) + ".tif"))

class MergedDataset(ArtificialNucleiDataset):

    def __init__(self,datasets):
        super(MergedDataset, self).__init__(self)
        self.image_path = []
        self.mask_path = []
        self.add_class("ArtificialNuclei", 1, 'ArtificialNuclei')
        image_path_train = []
        image_path_val = []
        mask_path_train = []
        mask_path_val = []
        self.ids = []
        run = 0
        for dataset in datasets:
            self.image_path.extend(dataset.image_path[0:dataset.train_cnt])
            self.mask_path.extend(dataset.mask_path[0:dataset.train_cnt])
           # self.ids.extend(dataset.ids[0:dataset.train_cnt]+self.ids.__len__())
        self.train_cnt = self.image_path.__len__()
        for dataset in datasets:
            self.image_path.extend(dataset.image_path[dataset.train_cnt:])
            self.mask_path.extend(dataset.mask_path[dataset.train_cnt:])
        self.val_cnt = self.image_path.__len__() - self.train_cnt
        ids_train = np.arange(0,self.train_cnt)
        ids_val = np.arange(self.train_cnt, self.train_cnt+self.val_cnt)
        np.random.shuffle(ids_train)
        np.random.shuffle(ids_val)
        self.ids = np.concatenate((ids_train,ids_val),axis=0)


    def load_data(self, width=256, height=256, ids=None, mode=1):
        for i in self.ids:
            self.add_image("ArtificialNuclei", image_id=i, path=None, width=width, height=height)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        img_final = cv2.imread(self.image_path[self.ids[image_id]])
        try:
            img_final = img_final[:,:,0]
        except:
            None
        #return img_final / 255.0
        try:
            img_final = img_final[:,0:256]
        except:
            e=1
        if self.settings.network_info["netinfo"] == 'maskrcnn': # mask rcnn need an rgb image
            img_new = np.zeros((img_final.shape[0],img_final.shape[1],3))
            img_new[:,:,0] = img_new[:,:,1] = img_new[:,:,2] = img_final
            img_final = img_new
        return img_final

class ArtificialNucleiDatasetNotConverted(ArtificialNucleiDataset):
    img_prefix = 'Img_'
    img_postfix = '.jpg' #'-inputs.png'
    mask_prefix = "Mask_"

    def setImagePaths(self, folders=""):
        for folder in folders:
            #self.img_prefix = os.path.basename(folder) + "_"
            folder_names = folder.split('\\')
            file_pattern = os.path.join(folder, self.img_prefix + "*" + self.img_postfix) #"Img_*-outputs.png")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            img_range = range(0,img_files.__len__())
            for i in img_range:
                #self.image_path.append(os.path.join(folder, "Img_" + str(i) + "-outputs.png"))
                self.image_path.append(os.path.join(folder, self.img_prefix + str(i) + self.img_postfix))
            # for i in img_files:
            #    self.image_path.append(i)
            return img_range

    def setMaskPaths(self, folders="",img_range=None):
        self.mask_postfix = ".tif"
        for folder in folders:
            #self.mask_prefix = os.path.basename(folder) + "_"
            #self.mask_prefix = "Mask_"
            folder_names = folder.split('\\')

            file_pattern = os.path.join(folder, self.mask_prefix + "*" + self.mask_postfix) #"Mask_*.tif")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            #for i in range(0,img_files.__len__()):
            for i in img_range:
                self.mask_path.append(os.path.join(folder, self.mask_prefix + str(i) + self.mask_postfix))
                #self.mask_path.append(os.path.join(folder, "Mask_" + str(i) + ".tif"))

    def load_image(self, image_id):
        info = self.image_info[image_id]
        img_final = cv2.imread(self.image_path[self.ids[image_id]])
        try:
            img_final = img_final[:,:,0]
        except:
            None
        #return img_final / 255.0
        img_final = img_final[:,0:256]
        if self.settings.network_info["netinfo"] == 'maskrcnn': # mask rcnn need an rgb image
            img_new = np.zeros((img_final.shape[0],img_final.shape[1],3))
            img_new[:,:,0] = img_new[:,:,1] = img_new[:,:,2] = img_final
            img_final = img_new
        return img_final

    def load_mask_one_layer(self, image_id,relabel=False):
        mask = tifffile.imread(self.mask_path[self.ids[image_id]])
        if (mask.ndim > 2):
            mask = mask[:,:,0]
        #mask = mask[:, 0:256]
        if (relabel):
            mask_tmp = np.zeros((mask.shape[0],mask.shape[1]))
            running=1
            for i in np.unique(mask):
                if i > 0:
                    mask_tmp = mask_tmp + running * (mask==i)
                    running = running + 1
            mask = mask_tmp.astype(np.float)
        return mask #mask.astype(np.float)

class SampleInference(ArtificialNucleiDataset):

    def setImagePaths(self, folders=""):
        self.img_postfix = ".jpg"
        for folder in folders:
            #self.img_prefix = os.path.basename(folder) + "_"
            folder_names = folder.split('\\')
            self.img_prefix = "Img"
            file_pattern = os.path.join(folder, self.img_prefix + "*" + self.img_postfix) #"Img_*-outputs.png")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            img_range = range(0,img_files.__len__())
            for i in img_range:
                #self.image_path.append(os.path.join(folder, "Img_" + str(i) + "-outputs.png"))
                self.image_path.append(img_files[i])
            # for i in img_files:
            #    self.image_path.append(i)
            return img_range

    def setMaskPaths(self, folders="",img_range=None):
        self.mask_postfix = ".tif"
        for folder in folders:
            #self.mask_prefix = os.path.basename(folder) + "_"
            #self.mask_prefix = "Mask_"
            folder_names = folder.split('\\')
            self.mask_prefix = "Img"
            file_pattern = os.path.join(folder, self.mask_prefix + "*" + self.mask_postfix) #"Mask_*.tif")
            print(file_pattern)
            img_files = glob.glob(file_pattern)
            img_files.sort()
            #for i in range(0,img_files.__len__()):
            for i in img_range:
                self.mask_path.append(img_files[i])
                #self.mask_path.append(os.path.join(folder, "Mask_" + str(i) + ".tif"))

    def load_data(self, width=256, height=256, ids=None, mode=1):
        # Load settings
        self.image_path = []
        self.mask_path = []

        self.add_class("ArtificialNuclei", 1, 'ArtificialNuclei')
        train_cnt = 0
        val_cnt = 0
        print("Loading train data ...")
        for i in self.settings.network_info["dataset_dirs_test"].split(';'):
            img_range = self.setImagePaths(folders=[i + "\\images"])
            self.setMaskPaths(folders=[i + "\\masks"],img_range=img_range)
        print("Checking train path ...")
        self.checkPath()
        self.ids = np.arange(0,self.image_path.__len__())
        for i in self.ids:
            self.add_image("ArtificialNuclei", image_id=i, path=None, width=width, height=height)
        return ids
