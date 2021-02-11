:: Pipeline used for the manuscript 'Evaluation of Deep Learning architectures forcomplex immunofluorescence nuclear imagesegmentation'
:: Florian Kromp, Lukas Fischer
:: 10.02.2021 
:: The pipeline aims at training and evaluating deep learning architectures on an annotated fluorescence nuclear image dataset

:: ------------------------------------ 0. Base Configuration ------------------------------------

:: Set path to the python environment and the code of the Generative adversarial network used to transform artificial image patches into natural-looking images
SET PYTHON_PIX2PIX=C:\ProgramData\Anaconda3\envs\tensorflownew\python.exe
SET PATH_PIX2PIX=E:\NuclearSegmentationPipeline\DeepLearningArchitectures\pix2pix-tensorflow-master\

:: Set path to scripts  used
SET PIPELINE_BASE_PATH=E:\NuclearSegmentationPipeline
SET PATH_DATASET_CONFIG=%PIPELINE_BASE_PATH%\Config\

:: Set path to the python environment and the scripts used to process the dataset, generate the synthetic images and create quantitative results
SET PYTHON_DATAGEN=C:\ProgramData\Anaconda3new\envs\tensorflownew\python.exe
SET PATH_DATAGEN=%PIPELINE_BASE_PATH%\DataGenerator\
SET PYTHON_RESULTS=C:\ProgramData\Anaconda3new\envs\tensorflownew\python.exe

:: Set all options the pipeline will run
SET DIAGNOSIS=normal Neuroblastoma Ganglioneuroma
SET DIAGNOSIS_TEST=normal Neuroblastoma Ganglioneuroma
SET MAX_EPOCHS_PIX2PIX=5
SET MAX_EPOCHS_SEGMENTATIONARCHITECTURE=5

:: Set the number of synthetic image patches to be created for the training and the validation set
SET NR_IMAGES=100

:: Set the overlap between the patches when tiling an image for the training/validation and the test set; the higher the overlap, the more patches are generated
SET OVERLAP_TRAINVAL=20
SET OVERLAP_TEST=50

SET DO_TRAINING=0
SET DO_EVALUATION=1

SET RETRAIN_PIX2PIX=1
SET COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING=0

:: Set if images shall be scaled to the same mean nuclear size before tiling; if yes (1), images are rescaled such that all images have the same mean nuclear size (area) across all images; if not (0), nuclei in images are not rescaled
SET SCALE_IMAGES=1

IF %SCALE_IMAGES%==1 (
	SET SCALE_TEXT=scaled
) ELSE (
	SET SCALE_TEXT=notscaled
)

SET TILE_FOLDER=tiles_%SCALE_TEXT%
SET RESULTS_FILE_PATH=%PIPELINE_BASE_PATH%\Results\Results_%SCALE_TEXT%.csv

:: ------------------------------------ 1. Configuration of architectures evaluated ------------------------------------

:: Set path to the python environment and the code of the deep learning environment(s) to be evaluated
@echo off
setlocal enabledelayedexpansion
set ARCHITECTURES=2

SET ARCHITECTURENAME[1]=classicunet
SET ARCHITECTURENAME[2]=maskrcnn

SET PYTHON_PATH_ARCHITECTURE[1]=C:\Users\florian.kromp\Downloads\WinPython-64bit-2.7.10.3\python-2.7.10.amd64\python.exe
SET PYTHON_PATH_ARCHITECTURE[2]=C:\ProgramData\Anaconda3\envs\maskrcnn\python.exe

SET CODE_PATH_ARCHITECTURE[1]=%PIPELINE_BASE_PATH%\DeepLearningArchitectures\UNet\
SET CODE_PATH_ARCHITECTURE[2]=%PIPELINE_BASE_PATH%\DeepLearningArchitectures\MaskRCNN\

SET TRAIN_SCRIPT[1]=train.py
SET TRAIN_SCRIPT[2]=train.py

SET EVAL_SCRIPT[1]=eval.py
SET EVAL_SCRIPT[2]=test.py
:: ------------------------------------ 2. Create image tiles (patches) from the annotated dataset ------------------------------------

:: Create folderpath for Testset
@echo off
setlocal enabledelayedexpansion
SET DATASET_FOLDER_PATH_STR_TEST= 
for %%i in (%DIAGNOSIS_TEST%) do (
	SET DATASET_FOLDER_PATH_STR_TEST=!DATASET_FOLDER_PATH_STR_TEST!%PATH_DATAGEN%%TILE_FOLDER%\test\%%i;
	)
@echo on

::Create tiles (patches) for training, validation and test set
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER% MKDIR %PATH_DATAGEN%%TILE_FOLDER%
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\train MKDIR %PATH_DATAGEN%%TILE_FOLDER%\train
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\val MKDIR %PATH_DATAGEN%%TILE_FOLDER%\val
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\val MKDIR %PATH_DATAGEN%%TILE_FOLDER%\test

:: Clean Results
copy NUL %RESULTS_FILE_PATH%

:: Create tiles for training and validation set 
for %%i in (%DIAGNOSIS%) do (
	IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\train\%%i\images MKDIR %PATH_DATAGEN%%TILE_FOLDER%\train\%%i\images
	IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\train\%%i\masks MKDIR %PATH_DATAGEN%%TILE_FOLDER%\train\%%i\masks
	IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\val\%%i\images MKDIR %PATH_DATAGEN%%TILE_FOLDER%\val\%%i\images
	IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\val\%%i\masks MKDIR %PATH_DATAGEN%%TILE_FOLDER%\val\%%i\masks
	IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\test\%%i\images MKDIR %PATH_DATAGEN%%TILE_FOLDER%\test\%%i\images
	IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\test\%%i\masks MKDIR %PATH_DATAGEN%%TILE_FOLDER%\test\%%i\masks
	%PYTHON_DATAGEN% %PATH_DATAGEN%run_createNaturalTiles.py --tissue %%i --outputFolder %PATH_DATAGEN%%TILE_FOLDER%\train\%%i --inputFolder %PATH_DATAGEN%dataset_train_val_test\train --scale %SCALE_IMAGES% --mode train --overlap %OVERLAP_TRAINVAL%
	%PYTHON_DATAGEN% %PATH_DATAGEN%run_createNaturalTiles.py --tissue %%i --outputFolder %PATH_DATAGEN%%TILE_FOLDER%\val\%%i --inputFolder %PATH_DATAGEN%dataset_train_val_test\val --scale %SCALE_IMAGES% --mode train --overlap %OVERLAP_TRAINVAL%
	%PYTHON_DATAGEN% %PATH_DATAGEN%run_createNaturalTiles.py --tissue %%i --outputFolder %PATH_DATAGEN%%TILE_FOLDER%\test\%%i --inputFolder %PATH_DATAGEN%dataset_train_val_test\test --scale %SCALE_IMAGES% --mode test --resultsfile %RESULTS_FILE_PATH% --overlap %OVERLAP_TEST%
)

:: ------------------------------------ 3. Train the GAN on paris of natural and synthetic images ------------------------------------

:: Train pix2pix GAN on paris of natural/artificial images
IF %RETRAIN_PIX2PIX%==1 (

	
	:: Create Pix2Pix paired datasets for each diagnosis
	for %%i in (%DIAGNOSIS%) do (
		IF NOT EXIST %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i MKDIR %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i
		%PYTHON_DATAGEN% %PATH_DATAGEN%run_createImagePairs.py --tissue %%i --outputFolder %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i --inputFolder %PATH_DATAGEN%dataset_train_val_test\train_and_val --scale %SCALE_IMAGES%
	)
	
	IF %COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING%==1 (
		:: Copy Pix2Pix paired datasets for combined diagnosis
		IF NOT EXIST %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\combined_diagnosis MKDIR %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\combined_diagnosis
		for %%i in (%DIAGNOSIS%) do (
			copy %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i\*.* %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\combined_diagnosis
		)
	)
	
	IF %COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING%==1 (
		:: Train Pix2Pix for combined dataset
		%PYTHON_PIX2PIX% %PATH_PIX2PIX%pix2pix.py --mode train --output_dir %PATH_PIX2PIX%checkpoints\checkpoint_combined_diagnosis_%SCALE_TEXT%  --max_epochs %MAX_EPOCHS_PIX2PIX% --input_dir %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\combined_diagnosis --which_direction BtoA
	)

	if %COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING%==0 (
		:: Train Pix2Pix for each dataset
		for %%i in (%DIAGNOSIS%) do (
			%PYTHON_PIX2PIX% %PATH_PIX2PIX%pix2pix.py --mode train --output_dir %PATH_PIX2PIX%checkpoints\checkpoint_%%i_%SCALE_TEXT% --max_epochs %MAX_EPOCHS_PIX2PIX% --input_dir %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i --which_direction BtoA
		)
	)
)

:: ------------------------------------ 4. Create synthetic images ------------------------------------

:: Create artificial dataset for each diagnosis present in the dataset
for %%i in (%DIAGNOSIS%) do (
	IF NOT EXIST %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\images MKDIR %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\images
	IF NOT EXIST %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\masks MKDIR %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\masks
	%PYTHON_DATAGEN% %PATH_DATAGEN%run_createArtificialDataset.py --tissue %%i --outputFolder %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\ --nr_images %NR_IMAGES% --overlapProbability 0.7 --inputFolder %PATH_DATAGEN%dataset_train_val_test\train  --scale %SCALE_IMAGES% 
)

:: ------------------------------------ 5. Infer synthetic images to natural-looking images using the trained GAN  ------------------------------------

:: Convert artificial dataset to natural-looking images for each diagnosis
if %COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING%==0 (
	for %%i in (%DIAGNOSIS%) do (
		IF NOT EXIST %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\train MKDIR %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\train
		IF NOT EXIST %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\val MKDIR %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\val
		%PYTHON_PIX2PIX% %PATH_PIX2PIX%pix2pix.py --mode test --output_dir %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i --input_dir %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\images --checkpoint %PATH_PIX2PIX%checkpoints\checkpoint_%%i_%SCALE_TEXT%
		IF NOT EXIST %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i\masks MKDIR %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i\masks
		copy %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\masks\*.tif %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i\masks\
	)
)

if %COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING%==1 (
	for %%i in (%DIAGNOSIS%) do (
		IF NOT EXIST %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\train MKDIR %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\train
		%PYTHON_PIX2PIX% %PATH_PIX2PIX%pix2pix.py --mode test --output_dir %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\combinedtrained_%%i --input_dir %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\images --checkpoint %PATH_PIX2PIX%checkpoints\checkpoint_combined_diagnosis_%SCALE_TEXT%
		IF NOT EXIST %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\combinedtrained_%%i\masks MKDIR %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\combinedtrained_%%i\masks
		copy %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\masks\*.tif %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\combinedtrained_%%i\masks\
	)
)

:: ------------------------------------ 6. Create pathes to the dataset ------------------------------------
:: Build string for training using all folders
@echo off
setlocal enabledelayedexpansion
SET DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL= 
SET DATASET_FOLDER_PATH_STR_VAL= 
SET DATASET_FOLDER_PATH_STR_TRAIN_NATURALONLY=
:: Add path to synthetic dataset to the training set path
if %COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING%==0 (
	for %%i in (%DIAGNOSIS%) do (
		SET DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL=!DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL!%PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i;
	)
)
if %COMBINE_DIAGNOSIS_FOR_PIX2PIX_TRAINING%==1 (
	for %%i in (%DIAGNOSIS%) do (
		SET DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL=!DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL!%PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\combinedtrained_%%i;
	)
)
@echo on

:: Add path to the natural dataset to the training set pathes and the validation set path
for %%i in (%DIAGNOSIS%) do (
	SET DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL=!DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL!%PATH_DATAGEN%%TILE_FOLDER%\train\%%i;
	SET DATASET_FOLDER_PATH_STR_TRAIN_NATURALONLY=!DATASET_FOLDER_PATH_STR_TRAIN_NATURALONLY!%PATH_DATAGEN%%TILE_FOLDER%\train\%%i;
	SET DATASET_FOLDER_PATH_STR_VAL=!DATASET_FOLDER_PATH_STR_VAL!%PATH_DATAGEN%%TILE_FOLDER%\val\%%i;
)

:: ------------------------------------ 7. Train and infer the deep network on natural and synthetic image patches of the training/validation set, vs. on natural image patches of the training/validation set only, infer predictions on the test set and reassemble the tiles to obtain the final predictions ------------------------------------
for %%n in (%ARCHITECTURES%) do ( 
	
	:: Train the architecture using synthetic and natural images
	IF %DO_TRAINING%==1 (
		:: Set settings for training using the artificial and natural image patches
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description !ARCHITECTURENAME[%%n]!_artificialAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES% --dataset mergeTisquantArtificial --max_epochs %MAX_EPOCHS_SEGMENTATIONARCHITECTURE% --dataset_dirs_train %DATASET_FOLDER_PATH_STR_TRAIN_ARTIFICIALANDNATURAL% --dataset_dirs_val %DATASET_FOLDER_PATH_STR_VAL%  --netinfo !ARCHITECTURENAME[%%n]! --traintestmode train

		::Train on artificial and natural image patches
		!PYTHON_PATH_ARCHITECTURE[%%n]! !CODE_PATH_ARCHITECTURE[%%n]!\!TRAIN_SCRIPT[%%n]!	

		:: Set settings for training using the natural image patches only
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description !ARCHITECTURENAME[%%n]!_naturalNuclei_dataset_%SCALE_TEXT% --dataset tisquant --max_epochs %MAX_EPOCHS_SEGMENTATIONARCHITECTURE% --dataset_dirs_train %DATASET_FOLDER_PATH_STR_TRAIN_NATURALONLY% --dataset_dirs_val %DATASET_FOLDER_PATH_STR_VAL% --netinfo !ARCHITECTURENAME[%%n]! --traintestmode train
		::Train on natural image patches only
		!PYTHON_PATH_ARCHITECTURE[%%n]! !CODE_PATH_ARCHITECTURE[%%n]!\!TRAIN_SCRIPT[%%n]!	
	)

	:: Infer the trained architecture on the test set
	IF %DO_EVALUATION%==1 (

		:: Remove old prediction
		del %PIPELINE_BASE_PATH%\Results\!ARCHITECTURENAME[%%n]!_artificialAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_predictions.h5
		del %PIPELINE_BASE_PATH%\Results\!ARCHITECTURENAME[%%n]!_artificialAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_predictions_reconstructed.pkl
		:: Set settings for Net Prediction
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description !ARCHITECTURENAME[%%n]!_artificialAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES% --dataset tisquant  --results_folder %PIPELINE_BASE_PATH%\Results --traintestmode test --netinfo !ARCHITECTURENAME[%%n]! --dataset_dirs_test %DATASET_FOLDER_PATH_STR_TEST%
		:: Run TisQuant UNet
		!PYTHON_PATH_ARCHITECTURE[%%n]! !CODE_PATH_ARCHITECTURE[%%n]!\!EVAL_SCRIPT[%%n]!	

		:: Reconstruct nn	
		%PYTHON_DATAGEN% %PATH_DATAGEN%run_reconstructResultMasks.py --scale %SCALE_IMAGES% --resultfile %RESULTS_FILE_PATH% --predictionfile %PIPELINE_BASE_PATH%\Results\!ARCHITECTURENAME[%%n]!_artificialAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_predictions.h5 --net !ARCHITECTURENAME[%%n]! --overlap %OVERLAP_TEST%

		:: Remove old prediction
		del %PIPELINE_BASE_PATH%\Results\!ARCHITECTURENAME[%%n]!_naturalNuclei_dataset_%SCALE_TEXT%_predictions.h5
		del %PIPELINE_BASE_PATH%\Results\!ARCHITECTURENAME[%%n]!_naturalNuclei_dataset_%SCALE_TEXT%_predictions_reconstructed.pkl
		:: Set settings for Net Prediction
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description !ARCHITECTURENAME[%%n]!_naturalNuclei_dataset_%SCALE_TEXT% --dataset tisquant  --results_folder %PIPELINE_BASE_PATH%\Results --traintestmode test --netinfo !ARCHITECTURENAME[%%n]! --dataset_dirs_test %DATASET_FOLDER_PATH_STR_TEST%

		:: Run Architecture
		!PYTHON_PATH_ARCHITECTURE[%%n]! !CODE_PATH_ARCHITECTURE[%%n]!\!EVAL_SCRIPT[%%n]!
		:: Reconstruct results
		%PYTHON_DATAGEN% %PATH_DATAGEN%run_reconstructResultMasks.py --scale %SCALE_IMAGES% --resultfile %RESULTS_FILE_PATH% --predictionfile %PIPELINE_BASE_PATH%\Results\!ARCHITECTURENAME[%%n]!_naturalNuclei_dataset_%SCALE_TEXT%_predictions.h5 --net !ARCHITECTURENAME[%%n]! --overlap %OVERLAP_TEST%
	)
)

:: ------------------------------------ 8. Generate quantitative results ------------------------------------

%PYTHON_RESULTS% %PIPELINE_BASE_PATH%\Evaluation\evaluation_tmi_createquantitativeresults.py
%PYTHON_RESULTS% %PIPELINE_BASE_PATH%\Evaluation\evaluation_tmi_createsinglecellannotationresults.py
