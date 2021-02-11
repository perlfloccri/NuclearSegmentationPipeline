import os
import sys
import random
import math
sys.path.append(r'E:\NuclearSegmentationPipeline\Config')
from Datahandling.Datasets import TisquantDatasetNew,ArtificialNucleiDataset,ArtificialNucleiDatasetNotConverted,MergedDataset, TisquantDataset, SampleInference
from Config.Config import UNETSettings
import re
import time
import numpy as np
import cv2
import matplotlib
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import skimage.transform as ski_transform
from skimage.measure import label
from utils import compute_final_mask, compute_final_mask2
from config import Config
import utils
import model as modellib
import visualize
from model import log

input_shape = [3, 256, 256]

# Root directory of the project
ROOT_DIR = r"E:\NuclearSegmentationPipeline\DeepLearningArchitectures\MaskRCNN"#os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class NucleiConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    #NAME = "Nuclei"
    NAME = UNETSettings().network_info["net_description"]
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + Nucleus

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    DETECTION_MIN_CONFIDENCE = 0.85 # before: 0.7

    # NEW Spot Config - delete afterwards
   # RPN_ANCHOR_SCALES = (4, 8, 16, 32)  # -32
   # BACKBONE_STRIDES = [2, 4, 8, 16] # -16
   # RPN_NMS_THRESHOLD = 0.5
   # DETECTION_MIN_CONFIDENCE = 0.5
   # TRAIN_ROIS_PER_IMAGE = 200
   # DETECTION_MAX_INSTANCES = 3000

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
"""
# Training dataset
dataset_test = TisquantDataset()
dataset_test.load_data(width=inference_config.IMAGE_SHAPE[0], height=inference_config.IMAGE_SHAPE[1],mode=2)
dataset_test.prepare()
"""

# Load Dataset
print ("Load dataset ...")
if UNETSettings().network_info["dataset"] == 'tisquant': #args.dataset
    dataset_test= TisquantDatasetNew()
elif UNETSettings().network_info["dataset"] == 'sampleInference':
    dataset_test = SampleInference()
else:
    print('Dataset not valid')
    sys.exit("Error")
dataset_test.load_data(mode=1)
dataset_test.prepare()

for image_id in dataset_test.image_ids:
    print (str(image_id)+ "\n")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                            dataset_test.class_names, figsize=(8, 8))


results = model.detect([original_image], verbose=1)

r = results[0]
#visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
#                            dataset_test.class_names, r['scores'], ax=get_ax())
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
#image_ids = np.random.choice(dataset_test.image_ids, 10)
APs = []
res = []
for image_id in dataset_test.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    erg_mask = utils.compute_final_mask2(r['masks'],r['scores'])
    #erg_old = utils.compute_final_mask(r['masks'])
    res.append(erg_mask)

print("mAP: ", np.mean(APs))

net_description = UNETSettings().network_info["net_description"]
net_result_folder = UNETSettings().network_info["results_folder"]
f = h5py.File(os.path.join(net_result_folder, net_description + '_predictions.h5'), 'a')
f.create_dataset('predictions', data=res, dtype=np.float32)
f.close()

"""
res = np.asarray(res, dtype=np.float32)
res = np.transpose(res, (1, 2, 0))
savemat((DATASET_FOLDER + '\\Evaluation1_prediction_MaskRCNN_testdata.mat'), mdict={'results_maskrcnn': res})  # Save generated predictions for the combined Net training
"""