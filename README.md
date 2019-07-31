# NuclearSegmentationPipeline

This repository contains the code for the paper

**Deep Learning architectures for generalized immunofluorescence based nuclear image segmentation**
<br>
[Florian Kromp](http://science.ccri.at/contact-us/contact-details/), [Lukas Fischer](https://www.scch.at/en/team/person_id/207), [Eva Bozsaky](http://science.ccri.at/contact-us/contact-details/), [Inge Ambros](http://science.ccri.at/contact-us/contact-details/), Wolfgang Doerr, [Sabine Taschner-Mandl](http://science.ccri.at/contact-us/contact-details/), [Peter Ambros](http://science.ccri.at/contact-us/contact-details/), Allan Hanbury

## Citing this work

If you use the Nuclear Segmentation Pipeline in your research, please cite the following BibTeX entry:

```
@article{kromp2019,
    author={Kromp, Florian and Fischer, Lukas and Bozsaky, Eva and Ambros, Inge and Doerr, Wolfgang and Taschner-Mandl, Sabine and Ambros, Peter and Hanbury, Allen},
    title={Deep Learning architectures for generalized immunofluorescence based nuclear image segmentation},
    journal = {arXiv},
    volume = {},
    number = {},
    pages = {},
    year = {2019},
    doi = {},
    note ={},
    URL = {https://arxiv.org/abs/1907.12975v1},
    eprint = {}
}
```

## Setup
Within this repository we provide a sample pipeline to utilize the U-Net architecture for nuclear image segmentation. The pipeline can be enhanced to work on other segmentation architectures as well,
we will update the repository including Mask R-CNN for segmentation in the near future.
The Nuclear Segmentation Pipeline was developed using a windows batch script. The batch script is necessary as we used multiple frameworks, each utilizing different python environments.
Therefore, the python environments must be set up using the requirement scripts provided in the respective folders ([DataGenerator](DataGenerator), [UnetPure](UnetPure), [lasagne_wrapper](lasagne_wrapper), [Pix2Pix GAN](pix2pix-tensorflow-master)). 

The following three environments have to be set: 
1. DataGenerator 
2. pix2pix GAN image translation 
3. U-net architecture + lasagne wrapper for the U-net architecture

After setting up the environments using the requirement files ([DataGenerator\requirements.txt](DataGenerator/requirements.txt), [UnetPure\requirements.txt](UnetPure/requirements.txt), [pix2pix-tensorflow-master\requirements.txt](pix2pix-tensorflow-master/requirements.txt)), the path to the python environments and the base path has to be set in the pipeline-batch-script ([run_pipeline_unet.bat](run_pipeline_unet.bat)).

Subsequently, the lasagne wrapper has to be built by activating the environment for the U-Net (UnetPure) and running these commands:
```
python setup.py build
python setup.py install
```

As we run all scripts using a batch script, absolute paths have to be set in the respective files: 
* [Config/Config/Config.py](Config/Config/Config.py): the config_path
* [DataGenerator/Classes/Helper.py](DataGenerator/Classes/Helper.py): the path to TileImages folder within the Config folder
* [Results/evaluation.py](Results/evaluation.py): the path to the DataGenerator folder
* [UnetPure/train.py](UnetPure/train.py), [UnetPure/eval.py](UnetPure/eval.py), [UNetPure/utils/cell_data.py](UnetPure/utils/cell_data.py): the path to the Config folder
* [UNetPure/models/unet1_augment.py](UnetPure/models/unet1_augment.py): the path to the Config folder

## Run 
Within the DataGenerator folder, there is the folder [dataset_train_val_test](DataGenerator/dataset_train_val_test) containing the dataset, with one subfolder for each diagnosis available.
Within each diagnosis folder, there are four folders: [train](DataGenerator/dataset_train_val_test/train), [val](DataGenerator/dataset_train_val_test/val), [train_and_val](DataGenerator/dataset_train_val_test/train_and_val) and [test](DataGenerator/dataset_train_val_test/test). The folder train_and_val contains all data of train + val.
Each of the folders contains a folder *images* and a folder *masks*, where raw images and annotations are stored (naming scheme is provided).
After putting your data there, you can run the batch script on a windows command line.
At success, the output (predictions of images of the test set) is stored in the [Results](Results) folder.

## Pipeline
The pipeline builds up upon multiple steps.
First, all images of the respective folders (train, val, test) are split in overlapping tiles (patches).
Then, a pix2pix architecture is trained on paired image patches (see description in the paper referenced above). The architecture learns to transform artifiically synthesized images into natural-like images.
Next, artificial image patches are synthesized based on the train- and val-data and pix2pix is used to transform them into natural-like image patches.
Finally, the respective segmentation architecture is train and evaluated on three conditions: 
- the artificial image patches only
- the artificial + natural image patches
- the natural image patches only
<p>Resulting predictions are inferred on test set patches and reassembled to obtain original image sizes.
They can be compared to groundtruth masks subsequently.
