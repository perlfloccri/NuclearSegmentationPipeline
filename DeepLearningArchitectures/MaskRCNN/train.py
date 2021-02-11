import os
import sys
sys.path.append(r'E:\NuclearSegmentationPipeline\Config')
from Datahandling.Datasets import TisquantDatasetNew,ArtificialNucleiDataset,ArtificialNucleiDatasetNotConverted,MergedDataset, TisquantDataset
from Config.Config import UNETSettings
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import skimage.transform as ski_transform
from skimage.measure import label
from utils import compute_final_mask
from config import Config
import utils
import model as modellib
import visualize
from model import log
#from Datasets import TisquantDataset

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
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, 192)
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    #BACKBONE_STRIDES = [4, 8, 16, 32, 64, 128]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = NucleiConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


"""
# Training dataset
dataset_train = TisquantDataset()
val_idx = dataset_train.load_data(width=config.IMAGE_SHAPE[0], height=config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = TisquantDataset()
dataset_val.load_data(width=config.IMAGE_SHAPE[0], height=config.IMAGE_SHAPE[1],ids=val_idx,mode=2)
dataset_val.prepare()
"""
# Load Dataset
print ("Load dataset ...")
if UNETSettings().network_info["dataset"] == 'tisquant': #args.dataset
    dataset= TisquantDatasetNew()
elif UNETSettings().network_info["dataset"] == 'artificialNuclei':
    dataset = ArtificialNucleiDataset()
elif UNETSettings().network_info["dataset"] == 'artificialNucleiNotConverted':
    dataset = ArtificialNucleiDatasetNotConverted()
elif UNETSettings().network_info["dataset"] == 'mergeTisquantArtificial':
    datasets = []
    dataset1 = TisquantDatasetNew()
    dataset1.load_data(mode=1)
    dataset2 = ArtificialNucleiDataset()
    dataset2.load_data(mode=1)
    datasets.append(dataset1)
    datasets.append(dataset2)
    dataset = MergedDataset(datasets)
else:
    print('Dataset not valid')
    sys.exit("Error")

# Load Dataset
dataset.load_data(mode=1)
dataset.prepare()

dataset_train, dataset_val = dataset.split_train_test()
images, labels = dataset_train.to_numpy()
test_images, test_labels = dataset_val.to_numpy()

"""
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
"""
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=UNETSettings().network_info["max_epochs"],
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=UNETSettings().network_info["max_epochs"],
            layers="all")
