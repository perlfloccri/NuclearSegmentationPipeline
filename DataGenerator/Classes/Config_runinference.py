class Config:
    diagnosis = ['Inference']
    magnification = ['63x','20x','10x']
    staining_type = ['Fluorescence']
    staining = ['Nucleus visualization']
    segmentation_function = 'nucleus'
    annotator = [1,20,23,11,12,16,17,19,20,23]#,19,20,23]
    device = 'FM'
    local_data_path = 'Z:\\'
    useBorderObjects = False
    outputPath = 'D:\DeepLearning\pix2pix-tensorflow-master\\'
    outputFolder = 'D:\DeepLearning\pix2pix-tensorflow-master\ArtToNat\\train\combined_tissues'
    scale=0
    mode='train'
    resultsfile = ''
    net=''
    overlap=20