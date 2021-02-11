# NuclearSegmentationPipeline

This repository contains the code for the paper

**Evaluation of Deep Learning architectures for complex immunofluorescence nuclear image segmentation**
<br>
[Florian Kromp](http://science.ccri.at/contact-us/contact-details/), [Lukas Fischer](https://www.scch.at/en/team/person_id/207), [Eva Bozsaky](http://science.ccri.at/contact-us/contact-details/), [Inge Ambros](http://science.ccri.at/contact-us/contact-details/), Wolfgang Doerr, Klaus Beiske, [Peter Ambros](http://science.ccri.at/contact-us/contact-details/), [Sabine Taschner-Mandl](http://science.ccri.at/contact-us/contact-details/) and [Allan Hanbury](http://allan.hanbury.eu/doku.php)

## Citing this work

If you use the Nuclear Segmentation Pipeline in your research, please cite the following BibTeX entry:

```
@article{Kromp2019,
    author={Kromp, Florian and Fischer, Lukas and Bozsaky, Eva and Ambros, Inge and Doerr, Wolfgang and Beiske, Klaus and Ambros, Peter, and Taschner-Mandl, Sabine and Hanbury, Allan},
    title={Evaluation of Deep Learning architectures for complex immunofluorescence nuclear image segmentation},
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
## Dataset used
```
@article{Kromp2020,
author = {Kromp, Florian and Bozsaky, Eva and Rifatbegovic, Fikret and Fischer, Lukas and Ambros, Magdalena and Berneder, Maria and Weiss, Tamara and Lazic, Daria and Doerr, Wolfgang and Hanbury, Allan and Beiske, Klaus and Ambros, Peter F. and Ambros, Inge M. and Taschner-Mandl, Sabine},
doi = {10.1038/s41597-020-00608-w},
journal = {Nature Scientific Data},
number = {262},
pages = {1--8},
title = {An annotated fluorescence image dataset for training nuclear segmentation methods},
volume = {7},
year = {2020}
}
```
The dataset can be accessed [here](https://identifiers.org/biostudies:S-BSST265).
## Architectures evaluated in the paper
U-Net: the code is available within this repository: [DeepLearningArchitectures/UNet](DeepLearningArchitectures/UNet)
<br>U-Net ResNet: the code is available at https://github.com/qubvel/segmentation_models
<br>Mask R-CNN: the code is available at https://github.com/matterport/Mask_RCNN
<br>KG Instance segmentation: the code is available at https://github.com/yijingru/KG_Instance_Segmentation
<br>Cellpose: the code is available at https://github.com/mouseland/cellpose

Attributed relational graphs: the code is available at https://github.com/sarslancs/argraphs and in the folder [ConventionalMethods/argraphs](ConventionalMethods/argraphs), where you can also find the code for parameter coarse- and finetuning
<br>Iterative h-min: the code is available at https://github.com/canfkoyuncu/IterativeHMinima and in the folder [ConventionalMethods/iterativehmin](ConventionalMethods/iterativehmin), where you can also find the code for parameter coarse- and finetuning

## Setup the pipeline using U-Net and Mask R-CNN
Within this repository we provide a sample pipeline to utilize the U-Net and the Mask R-CNN architectures for nuclear image segmentation. The pipeline can be enhanced to work on other segmentation architectures as well,
you can find instructions how to include any architecture [below](#add-architectures-to-the-pipeline).
The Nuclear Segmentation Pipeline was developed using a windows batch script. The batch script is necessary as we used multiple frameworks, each utilizing different python environments.
Therefore, the python environments must be set up using the requirement scripts provided in the respective folders ([DataGenerator](DataGenerator), [Unet](DeepLearningArchitectures/UNet), [lasagne_wrapper](DeepLearningArchitectures/lasagne_wrapper), [Pix2Pix GAN](DeepLearningArchitectures/pix2pix-tensorflow-master), [Mask R-CNN](DeepLearningArchitectures/MaskRCNN), [Result generation](Results)). 

The following environments have to be set in order to run the pipeline (if other architectures are used, the required environment has to be set up): 
1. DataGenerator 
2. pix2pix GAN image translation 
3. U-net architecture + lasagne wrapper for the U-net architecture
4. Mask R-CNN
5. Result generation
After setting up the environments, the path to the python environments, the code base and the base pipeline path have to be set in the pipeline-batch-script ([run_pipeline.bat](run_pipeline.bat)), see also the respective step in the [pipeline](#pipeline-to-train-and-infer-segmentation-architectures).

Subsequently, the lasagne wrapper has to be built by activating the environment for the U-Net (UnetPure) and running these commands:
```
python setup.py build
python setup.py install
```

As we run all scripts using a batch script, absolute paths have to be set in the respective files: 
* [Config/Config/Config.py](Config/Config/Config.py): the config_path
* [DataGenerator/Classes/Helper.py](DataGenerator/Classes/Helper.py): the path to TileImages folder within the Config folder
* [Evaluation/evaluation_tmi_createquantitativeresults.py](Evaluation/evaluation_tmi_createquantitativeresults.py): the path to the DataGenerator folder, the path to the csv-file in the results folder (dependent on the setting in the [pipeline](run_pipeline.bat)), and the path to the annotation description csv-file [DataGenerator/image_description_final_revision.csv](DataGenerator/image_description_final_revision.csv)
* [Evaluation/evaluation_tmi_createsinglecellannotationresults.py](Evaluation/evaluation_tmi_createsinglecellannotationresults.py): the path to the DataGenerator folder, the path to the csv-file in the results folder (dependent on the setting in the [pipeline](run_pipeline.bat)), and the path to the annotation description csv-file [DataGenerator/image_description_final_revision.csv](DataGenerator/image_description_final_revision.csv)
* [DeepLearningArchitectures/Unet/train.py](DeepLearningArchitectures/Unet/train.py), [DeepLearningArchitectures/Unet/eval.py](DeepLearningArchitectures/Unet/eval.py), [DeepLearningArchitectures/UNet/utils/cell_data.py](DeepLearningArchitectures/Unet/utils/cell_data.py): the path to the Config folder
* [DeepLearningArchitectures/UNet/models/unet1_augment.py](DeepLearningArchitectures/Unet/models/unet1_augment.py): the path to the Config folder
* [DeepLearningArchitectures/MaskRCNN/train.py](DeepLearningArchitectures/MaskRCNN/train.py), [DeepLearningArchitectures/MaskRCNN/test.py](DeepLearningArchitectures/MaskRCNN/test.py), 

## Run 
Within the [DataGenerator](DataGenerator) folder, there are two zip files (dataset_train_val_test.zip, dataset_singlecellgroundtruth.zip), extract them first (they contain a subsample of the dataset used to create the results for the publication, the entire dataset can be found [here](#dataset-used)). 
Upon extraction, there are two folders, the folder dataset_train_val_test containing the dataset, with one subfolder for each diagnosis available (there can be different diagnosis between train and val vs. the test set) and
the folder dataset_singlecellgroundtruth.
Within the dataset_train_val_test folder, there are four folders: train,val train_and_val and test. The folder train_and_val contains all data of train + val.
Each of the folders contains a folder *images* and a folder *masks*, where raw images and annotations are stored (naming scheme and sample data is provided).
The same structure except for leaving out the train, val, train_and_val, and test subfolders accounts for the folder dataset_singlecellgroundtruth, holding the single-cell annotations for the dataset.

Now you can run the batch script on a windows command line.
At success, the output (predictions of images of the test set) is stored in the [Results](Results) folder.

## Pipeline to train and infer segmentation architectures
The pipeline builds up upon multiple steps.
0. Base Configuration: general variables and pathes are set
	- DIAGNOSIS: diagnosis available for training and validation set (separated by blank)
	- DIAGNOSIS_TEST: diagnosis available for test set (separated by blank)
	- MAX_EPOCHS_PIX2PIX: number of epochs used to train pix2pix
	- MAX_EPOCHS_SEGMENTATIONARCHITECTURE: max. number of epochs used to train the deep learning architectures to be evaluated
	- NR_IMAGES: Number of synthetic images to be created
	- OVERLAP_TRAINVAL: overlap between patches while tiling images of the training- and validation set
	- OVERLAP_TEST: overlap between patches while tiling images of the test set
	- COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING: Switch deciding if all annotated images of a certain diagnosis shall be used to train the pix2pix in a combined mode (1), or separately per diagnosis (0); depending on the size of the annotated dataset of each diagnosis 
	- SCALE_IMAGES: Switch deciding if images shall be rescaled to the same mean nuclear size across images before tiling
1. Configuration of architectures evaluated
	- ARCHITECTURES: one continuous number for each architecture, starting from 1
	- ARCHITECTURE_NAMES: name of the architecture; steers the format of the dataset to be loaded for the respective architecture as defined in [Config][DataHandling][Datasets], Class DataLoading
	- PYTHON_PATH_ARCHITECTURE: path to the python environment for each architecture
	- CODE_PATH_ARCHITECTURE: path to the code for each architecture
	- TRAIN_SCRIPT: training script to be called for each architecture
	- EVAL_SCRIPT: inference script to be called for each architecture 
2. All images of the respective folders (train, val, test) are split in overlapping tiles (patches).
3. A pix2pix architecture is trained on paired image patches (see description in the paper referenced above). The pix2pix architecture learns to transform artifically synthesized images into natural-like images. Setting a switch in step 0 of the pipeline it can be decided if one GAN per diagnosis shall be trained or if all diagnosis(the entire dataset) is used to train one GAN (the choice depends on the size of the annotated datasets with respect to the diagnosis).
4. Artificial image patches are synthesized based on the train- and val-data
5. The trained pix2pix GAN is used to transform them into natural-like image patches.
6. Create pathes to the dataset
7. The architectures to be evaluated are trained on the natural and the artificial image patches of the training/validation set vs. the natural image patches of the training/validation set only, inferred on the test set and the predicted patches are finally reassembled to obtain the final predictions.
8. Generate quantitative results based on the predictions

# Evaluation scripts
All scripts to generate quantitative results from inferred predictions can be found in the [Evaluation](Evaluation) folder and are included in the last step (8) of the pipeline.

- evaluation_tmi_generatequantitativeresults.py
	- Calculation of segmentation metrics for all results with respect to the groundtruth annotation
	- All annotated targets available in [DataGenerator][image_description_final_revision.csv] can be used for evaluation, but have to be adapted in the class [Evaluation][Tools][StructuredEvaluation.py], during processing this file in [Evaluation][evaluation_tmi_generatequantitativeresults.py] and defined as target upon writing the results to the csv-file.
- evaluation_tmi_createsinglecellannotationresults.py
	- Calculation of segmentation metrics for all results with respect to the single cell groundtruth annotation
	- All annotated targets available in [DataGenerator][image_description_final_revision.csv] can be used for evaluation, but have to be adapted in the class [Evaluation][Tools][StructuredEvaluation.py], during processing this file in [Evaluation][evaluation_tmi_createsinglecellannotationresults.py] and defined as target upon writing the results to the csv-file.

Within this repository, only a subset of the dataset used to generate the results for the publication is provided.
The entire dataset can be downloaded from the BioStudies database as described in the beginning.

## Add architectures to the pipeline
To add and evaluate more architectures, first the training/inference scripts of the respective architecture have to be adapted.

	- Modify the training file of the respective architecture; Settings are modified by the definition in the batch script and imported, they do not need to be set here:
		sys.path.append(PATH_TO_DATAHANDLING_FOLDER) # Replace PATH_TO_DATAHANDLING_FOLDER by the path to the folder [Config][Datahandling]
		from Datahandling.Datasets import DataLoading
		
		### Load training dataset and split into training/validation set
		dataloader = DataLoading()
            dataset = dataloader.load()
            [dtrain,dval] = dataset.split_train_test()
		dataloader = DataLoading()
		
		### Save network weights during training including dataloader.getID() in the name, e.g. weight_path = 'model_' + dataloader.getID() + '.pth'
		
	- Modify the inference file of the respective architecture; Settings are modified by the definition in the batch script and imported, they do not need to be set here:
		sys.path.append(PATH_TO_DATAHANDLING_FOLDER) # Replace PATH_TO_DATAHANDLING_FOLDER by the path to the folder Datahandling
		from Datahandling.Datasets import DataLoading
        
		# Load network weights that were trained including dataloader.getID() in the name, e.g. weight_path = 'model_' + dataloader.getID() + '.pth'
		
		# Load the dataset
		dsets = dataloader.load(phase='test')
		
		### Iterate over all patches of the dataset 
		P = []
		for index in range(len(dsets)):
			patch = dsets.load_image(index)
			prediction = test_inference(args, patch) # replace by the respective method to infer results
			P.append(prediction) # append prediction to array
			
		# Save results in the prediction path
            f = h5py.File(os.path.join(dataloader.getResultsPath(), dataloader.getID() + "_predictions.h5"), 'a')
            f.create_dataset('predictions', data=P, dtype=np.float32)
            f.close()


Then, the pipeline has to be adapted to add the architecture to the configuration (Pipeline step 1) and to add variables pointing to the respective python environment/the architecture code: 

If actually two architectures are evaluated, add the third one like this:

	SET ARCHITECTURES=1 2 3
	SET ARCHITECTURE_NAMES[3]= CELLPOSE
	SET PYTHON_PATH_ARCHITECTURE[3] = PATH_TO_CELLPOSE_PYTHON_ENVIRONMENT
	SET CODE_PATH_ARCHITECTURE[3] = PATH_TO_CELLPOSE_CODE
	
If the architectures need the images in a specific format, the [DataHandling class in Config/Datahandling/Datasets.py](Config/Datahandling/Datasets.py) class must be adapted.