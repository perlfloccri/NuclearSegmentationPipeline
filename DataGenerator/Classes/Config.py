class Config:
    diagnosis = ['Ganglioneuroblastoma']#'Ganglioneuroma']#'Neuroblastoma', 'Neuroblastoma_cell_line', 'Ganglioneuroma', 'Ganglioneuroblastoma', 'normal']
    magnification = ['63x']#,'20x','10x']
    staining_type = ['Fluorescence']
    staining = ['Nucleus visualization']
    segmentation_function = 'nucleus'
    annotator = [1]#[34]#1,20,23,11,12,16,17,19,20,23]#,19,20,23]
    device = '3'#'FM'
    local_data_path = 'Z:\\'
    useBorderObjects = False
    outputPath = 'D:\DeepLearning\pix2pix-tensorflow-master\\'
    outputFolder = 'D:\DeepLearning\pix2pix-tensorflow-master\ArtToNat\\train\combined_tissues'
    scale=0
    mode='train'
    resultsfile = ''
    net=''
    overlap=20