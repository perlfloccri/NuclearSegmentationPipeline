class Config:
    diagnosis = ['normal', 'Neuroblastoma','Ganglioneuroma']
    magnification = ['63x','20x','10x']
    staining_type = ['Fluorescence']
    staining = ['Nucleus visualization']
    segmentation_function = 'nucleus'
    annotator = [1,20,23,11,12,16,17,19,20,23]#,19,20,23]
    device = 'FM'
    local_data_path = 'Z:\\'
    useBorderObjects = False
    outputPath = 'D:\DeepLearning\DataGenerator\\'
    outputFolder = 'D:\DeepLearning\DataGenerator\spot_gH2AX_allimages'
    scale=0
    mode='train'
    resultsfile = ''
    net=''
    overlap=20