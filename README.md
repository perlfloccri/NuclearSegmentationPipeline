# NuclearSegmentationPipeline

This repository contains the code for the paper

**Deep Learning architectures for generalized immunofluorescence based nuclear image segmentation**
<br>
[Florian Kromp](http://science.ccri.at/contact-us/contact-details/), [Lukas Fischer](https://www.scch.at/en/team/person_id/207), [Eva Bozsaky](http://science.ccri.at/contact-us/contact-details/), [Inge Ambros](http://science.ccri.at/contact-us/contact-details/), Wolfgang Doerr, [Sabine Taschner-Mandl](http://science.ccri.at/contact-us/contact-details/), [Peter Ambros](http://science.ccri.at/contact-us/contact-details/), Allan Hanbury

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

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
    URL = {},
    eprint = {}
}
```

## Setup
Within this repository we provide a sample pipeline to utilize the U-Net architecture for nuclear image segmentation. The pipeline can be enhanced to work on other segmentation architectures as well,
we will update the repository including Mask R-CNN for segmentation.
The Nuclear Segmentation Pipeline was developed using a windows batch script. The batch script is necessary as we used multiple frameworks, each utilizing different python environments.
Therefore, the python environments must be set using the requirement scripts provided. 
<br>The following three environments have to be set: 1. DataGenerator 2. pix2pix image translation 3. U-net architecture + lasagne wrapper for the U-net architecture
After setting up the environments using the requirement files (DataGenerator\requirements.txt, UnetPure\requirements.txt, pix2pix-tensorflow-master\requirements.txt), 
the path to the python environments and the base path has to be set in the pipeline-batch-script (run_pipeline_unet.bat).
<br>Subsequently, the lasagne wrapper has to be built: change to the environment for the U-Net (UnetPure) and run these commands:
```
python setup.py build
python setup.py install
```

As we run all scripts using a batch script, absolute pathes have to be set in the respective files: 
Config/Config/Config.py: the config_path
DataGenerator/Classes/Helper.py: the path to TileImages folder within the Config folder
Results/evaluation.py: the path to the DataGenerator folder
UnetPure/train.py, UnetPure/eval.py, UNetPure/utils/cell_data.py: the path to the Config folder
UNetPure/models/unet1_augment.py: the path to the Config folder

## Run 
Within the DataGenerator folder, there is the folder dataset_train_val_test containing the dataset, separated into train, val and test. 
Each of the folders contains a folder images and a folder masks, where raw images and annotations are stored (naming scheme is provided).
After putting your data there, you can run the batch script on a dos command line.
At success, the output (predictions of images of the test set) is stored in the Results folder.
<br>Please dont't forget to [cite](#citing-this-work)!