SET PYTHON_DATAGEN=C:\ProgramData\Anaconda3new\envs\tensorflownew\python.exe
SET PYTHON_PIX2PIX=C:\ProgramData\Anaconda3\envs\tensorflownew\python.exe
SET PYTHON_TISQUANT_UNET=C:\Users\florian.kromp\Downloads\WinPython-64bit-2.7.10.3\python-2.7.10.amd64\python.exe

SET PIPELINE_BASE_PATH=E:\github_publication
SET PATH_DATAGEN=%PIPELINE_BASE_PATH%\DataGenerator\
SET PATH_PIX2PIX=E:\github_publication\pix2pix-tensorflow-master\
SET PATH_DATASET_CONFIG=%PIPELINE_BASE_PATH%\Config\
SET PATH_UNET=%PIPELINE_BASE_PATH%\UnetPure\

SET DIAGNOSIS=normal
SET MAX_EPOCHS_PIX2PIX=5
SET MAX_EPOCHS_UNET=5

SET NR_IMAGES=100
SET NR_IMAGES_VAL=20
SET OVERLAP_TRAINVAL=20
SET OVERLAP_TEST=50

SET SCALE_IMAGES=1
SET DO_TRAINING=1
SET DO_EVALUATION=1

SET RETRAIN_PIX2PIX=1
SET DO_DB_EXPORT=0

IF %SCALE_IMAGES%==1 (
	SET SCALE_TEXT=scaled
) ELSE (
	SET SCALE_TEXT=notscaled
)
echo %SCALE_TEXT%

SET TILE_FOLDER=tiles_%SCALE_TEXT%
SET RESULTS_FILE_PATH=%PIPELINE_BASE_PATH%\Results\Results\results_%SCALE_TEXT%.csv
goto:artificial_only

:: Create folderpath for Testset
:creat_pattest
@echo off
setlocal enabledelayedexpansion
SET DATASET_FOLDER_PATH_STR_TEST= 
for %%i in (%DIAGNOSIS%) do (
	SET DATASET_FOLDER_PATH_STR_TEST=!DATASET_FOLDER_PATH_STR_TEST!%PATH_DATAGEN%%TILE_FOLDER%\test\%%i;
	)
@echo on

:: Create Training/Validation/Testset
IF %DO_DB_EXPORT%==1 (
	IF EXIST %PATH_DATAGEN%dataset_train_val_test rd /s /q %PATH_DATAGEN%dataset_train_val_test
	IF NOT EXIST %PATH_DATAGEN%dataset_train_val_test MKDIR %PATH_DATAGEN%dataset_train_val_test
	for %%i in (%DIAGNOSIS%) do (
		%PYTHON_DATAGEN% %PATH_DATAGEN%run_createDBExportSets.py --tissue %%i --outputFolder %PATH_DATAGEN%dataset_train_val_test
	)
)

::Create tiles
:create_tiles
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER% MKDIR %PATH_DATAGEN%%TILE_FOLDER%
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\train MKDIR %PATH_DATAGEN%%TILE_FOLDER%\train
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\val MKDIR %PATH_DATAGEN%%TILE_FOLDER%\val
IF NOT EXIST %PATH_DATAGEN%%TILE_FOLDER%\val MKDIR %PATH_DATAGEN%%TILE_FOLDER%\test

:: Clean Results
copy NUL %RESULTS_FILE_PATH%


for %%i in (%DIAGNOSIS%) do (
:: Create tiles for training and validation set 
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
goto:ende
:retrainpix2pix
IF %RETRAIN_PIX2PIX%==1 (

	
		:: Create Pix2Pix paired datasets for each diagnosis
		for %%i in (%DIAGNOSIS%) do (
			IF NOT EXIST %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i MKDIR %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i
			%PYTHON_DATAGEN% %PATH_DATAGEN%run_createImagePairs.py --tissue %%i --outputFolder %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i --inputFolder %PATH_DATAGEN%dataset_train_val_test\train_and_val --scale %SCALE_IMAGES%
		)

	:really_train
	:: Train Pix2Pix for each dataset
		for %%i in (%DIAGNOSIS%) do (
			%PYTHON_PIX2PIX% %PATH_PIX2PIX%pix2pix.py --mode train --output_dir %PATH_PIX2PIX%checkpoints\checkpoint_%%i_%SCALE_TEXT% --max_epochs %MAX_EPOCHS_PIX2PIX% --input_dir %PATH_DATAGEN%PairedImages\%SCALE_TEXT%\train\%%i --which_direction BtoA
		)
)
goto:ende
:create_artificial_images
IF NOT EXIST %PATH_DATAGEN%convertedPairedImages\%SCALE_TEXT% MKDIR %PATH_DATAGEN%convertedPairedImages\%SCALE_TEXT%

:: Create artificial dataset for each diagnosis
for %%i in (%DIAGNOSIS%) do (
	IF NOT EXIST %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\images MKDIR %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\images
	IF NOT EXIST %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\val\%%i\images MKDIR %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\val\%%i\images
	IF NOT EXIST %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\masks MKDIR %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\masks
	IF NOT EXIST %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\val\%%i\masks MKDIR %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\val\%%i\masks
	%PYTHON_DATAGEN% %PATH_DATAGEN%run_createArtificialDataset.py --tissue %%i --outputFolder %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\ --nr_images %NR_IMAGES% --overlapProbability 0.7 --inputFolder %PATH_DATAGEN%dataset_train_val_test\train  --scale %SCALE_IMAGES% 
	%PYTHON_DATAGEN% %PATH_DATAGEN%run_createArtificialDataset.py --tissue %%i --outputFolder %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\val\ --nr_images %NR_IMAGES_VAL% --overlapProbability 0.7 --inputFolder %PATH_DATAGEN%dataset_train_val_test\val  --scale %SCALE_IMAGES% 
)

:convert_art
:: Convert artificial dataset to natural images for each diagnosis
	for %%i in (%DIAGNOSIS%) do (
		IF NOT EXIST %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\train MKDIR %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\train
		IF NOT EXIST %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\val MKDIR %PATH_PIX2PIX%convertedArtificialImages\%SCALE_TEXT%\val
		%PYTHON_PIX2PIX% %PATH_PIX2PIX%pix2pix.py --mode test --output_dir %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i --input_dir %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\images --checkpoint %PATH_PIX2PIX%checkpoints\checkpoint_%%i_%SCALE_TEXT%
		%PYTHON_PIX2PIX% %PATH_PIX2PIX%pix2pix.py --mode test --output_dir %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\val\singletrained_%%i --input_dir %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\val\%%i\images --checkpoint %PATH_PIX2PIX%checkpoints\checkpoint_%%i_%SCALE_TEXT%
	)

:copy_images
:: Move resulting images and masks to new folder
for %%i in (%DIAGNOSIS%) do (
	IF NOT EXIST %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i\masks MKDIR %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i\masks
	IF NOT EXIST %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\val\singletrained_%%i\masks MKDIR %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\val\singletrained_%%i\masks
	copy %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\train\%%i\masks\*.tif %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i\masks\
	copy %PATH_DATAGEN%artificialImages\%SCALE_TEXT%\val\%%i\masks\*.tif %PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\val\singletrained_%%i\masks\
)
goto:ende
:artificial_only


:: Build string for training using all folders
@echo off
setlocal enabledelayedexpansion
SET DATASET_FOLDER_PATH_STR_TRAIN= 
SET DATASET_FOLDER_PATH_STR_VAL= 
for %%i in (%DIAGNOSIS%) do (
	SET DATASET_FOLDER_PATH_STR_TRAIN=!DATASET_FOLDER_PATH_STR_TRAIN!%PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\train\singletrained_%%i;
	SET DATASET_FOLDER_PATH_STR_VAL=!DATASET_FOLDER_PATH_STR_VAL!%PATH_DATAGEN%convertedArtificialImages\%SCALE_TEXT%\val\singletrained_%%i;
)
@echo on

	IF %DO_TRAINING%==1 (
		:: Set settings for Net Training
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description classicunet_artificialNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL% --dataset artificialNuclei --max_epochs %MAX_EPOCHS_UNET% --dataset_dirs_train %DATASET_FOLDER_PATH_STR_TRAIN% --dataset_dirs_val %DATASET_FOLDER_PATH_STR_VAL% --netinfo classicunet --traintestmode train

		::Train
		%PYTHON_TISQUANT_UNET% %PATH_UNET%train.py
	)
	IF %DO_EVALUATION%==1 (
		:: Remove old prediction
		del %PIPELINE_BASE_PATH%\Results\Results\classicunet_artificialNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL%_predictions.h5
		del %PIPELINE_BASE_PATH%\Results\Results\classicunet_artificialNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL%_predictions_reconstructed.pkl
		:: Set settings for Net Prediction
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description classicunet_artificialNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL% --dataset tisquant  --results_folder %PIPELINE_BASE_PATH%\Results\Results --traintestmode test --netinfo classicunet --dataset_dirs_test %DATASET_FOLDER_PATH_STR_TEST%
		:: Run TisQuant UNet
		%PYTHON_TISQUANT_UNET% %PATH_UNET%eval.py
		:: Reconstruct results
		%PYTHON_DATAGEN% %PATH_DATAGEN%run_reconstructResultMasks.py --scale %SCALE_IMAGES% --resultfile %RESULTS_FILE_PATH% --predictionfile %PIPELINE_BASE_PATH%\Results\Results\classicunet_artificialNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL%_predictions.h5 --net classicunet --overlap %OVERLAP_TEST%
	)


:artificial_and_natural
:: Start pipelines for the natural and artificial dataset
for %%i in (%DIAGNOSIS%) do (
	SET DATASET_FOLDER_PATH_STR_TRAIN=!DATASET_FOLDER_PATH_STR_TRAIN!%PATH_DATAGEN%%TILE_FOLDER%\train\%%i;
	SET DATASET_FOLDER_PATH_STR_VAL=!DATASET_FOLDER_PATH_STR_VAL!%PATH_DATAGEN%%TILE_FOLDER%\val\%%i;
	)

	IF %DO_TRAINING%==1 (
		:: Set settings for Net Training
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description classicunet_artificialNotConvertedAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL% --dataset mergeTisquantArtificial --max_epochs %MAX_EPOCHS_UNET% --dataset_dirs_train %DATASET_FOLDER_PATH_STR_TRAIN% --dataset_dirs_val %DATASET_FOLDER_PATH_STR_VAL%  --netinfo classicunet --traintestmode train

		::Train
		%PYTHON_TISQUANT_UNET% %PATH_UNET%train.py
	)
	IF %DO_EVALUATION%==1 (

		:: Remove old prediction
		del %PIPELINE_BASE_PATH%\Results\Results\classicunet_artificialNotConvertedAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL%_predictions.h5
		del %PIPELINE_BASE_PATH%\Results\Results\classicunet_artificialNotConvertedAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL%_predictions_reconstructed.pkl
		:: Set settings for Net Prediction
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description classicunet_artificialNotConvertedAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL% --dataset tisquant  --results_folder %PIPELINE_BASE_PATH%\Results\Results --traintestmode test --netinfo classicunet --dataset_dirs_test %DATASET_FOLDER_PATH_STR_TEST%
		:: Run TisQuant UNet
		%PYTHON_TISQUANT_UNET% %PATH_UNET%eval.py

		:: Reconstruct nn
		%PYTHON_DATAGEN% %PATH_DATAGEN%run_reconstructResultMasks.py --scale %SCALE_IMAGES% --resultfile %RESULTS_FILE_PATH% --predictionfile %PIPELINE_BASE_PATH%\Results\Results\classicunet_artificialNotConvertedAndNaturalNuclei_dataset_%SCALE_TEXT%_%NR_IMAGES%_%NR_IMAGES_VAL%_predictions.h5 --net classicunet --overlap %OVERLAP_TEST%
	)

:: Start pipelines for the natural only dataset
:natural_only

:: Build string for training using all folders
@echo off
setlocal enabledelayedexpansion
SET DATASET_FOLDER_PATH_STR_TRAIN= 
SET DATASET_FOLDER_PATH_STR_VAL= 
for %%i in (%DIAGNOSIS%) do (
	SET DATASET_FOLDER_PATH_STR_TRAIN=!DATASET_FOLDER_PATH_STR_TRAIN!%PATH_DATAGEN%%TILE_FOLDER%\train\%%i;
	SET DATASET_FOLDER_PATH_STR_VAL=!DATASET_FOLDER_PATH_STR_VAL!%PATH_DATAGEN%%TILE_FOLDER%\val\%%i;
	)
@echo on

	IF %DO_TRAINING%==1 (
		:: Set settings for Net Training
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description classicunet_naturalNuclei_dataset_%SCALE_TEXT% --dataset tisquant --max_epochs %MAX_EPOCHS_UNET% --dataset_dirs_train %DATASET_FOLDER_PATH_STR_TRAIN% --dataset_dirs_val %DATASET_FOLDER_PATH_STR_VAL% --netinfo classicunet --traintestmode train

		::Train
		%PYTHON_TISQUANT_UNET% %PATH_UNET%train.py
	)
	IF %DO_EVALUATION%==1 (
		:: Remove old prediction
		del %PIPELINE_BASE_PATH%\Results\classicunet_naturalNuclei_dataset_%SCALE_TEXT%_predictions.h5
		del %PIPELINE_BASE_PATH%\Results\classicunet_naturalNuclei_dataset_%SCALE_TEXT%_predictions_reconstructed.pkl
		:: Set settings for Net Prediction
		%PYTHON_DATAGEN% %PATH_DATASET_CONFIG%Config\Config.py --startup 0 --net_description classicunet_naturalNuclei_dataset_%SCALE_TEXT% --dataset tisquant  --results_folder %PIPELINE_BASE_PATH%\Results\Results --traintestmode test --netinfo classicunet --dataset_dirs_test %DATASET_FOLDER_PATH_STR_TEST%

		:: Run TisQuant UNet
		%PYTHON_TISQUANT_UNET% %PATH_UNET%eval.py
		:: Reconstruct results
		%PYTHON_DATAGEN% %PATH_DATAGEN%run_reconstructResultMasks.py --scale %SCALE_IMAGES% --resultfile %RESULTS_FILE_PATH% --predictionfile %PIPELINE_BASE_PATH%\Results\Results\classicunet_naturalNuclei_dataset_%SCALE_TEXT%_predictions.h5 --net classicunet --overlap %OVERLAP_TEST%
	)
:ende
