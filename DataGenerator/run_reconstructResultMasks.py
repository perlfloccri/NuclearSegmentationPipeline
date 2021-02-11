from Classes.Config import Config
from Classes.Helper import Tools
from Classes.Image import AnnotatedImage,AnnotatedObjectSet, ArtificialAnnotatedImage
from matplotlib import pyplot as plt
import os
import argparse
import glob
import pandas
import csv
import numpy as np
from Classes.Image import Image
from tifffile import imread
import h5py
import pickle

def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--scale', help='select output folder', default=None)
    parser.add_argument('--resultfile', help='select result file', default=None)
    parser.add_argument('--predictionfile', help='select result file', default=None)
    parser.add_argument('--net', help='describe net', default=None)
    parser.add_argument('--overlap', help='select output folder', default=None)
    args = parser.parse_args()
    config = Config

    if args.scale == '1':
        config.scale=True
    else:
        config.scale=False
    if args.resultfile:
        config.resultfile=args.resultfile
    else:
        print("No result file provided")
        exit()
    if args.net:
        config.net=args.net
    if args.overlap:
        config.overlap=int(args.overlap)

    path_to_img = []
    tiles = []
    images = []
    scales = []
    scales_new = []
    with open(args.resultfile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            path_to_img.append(row[0])
            scales.append(float(row[1]))
            tiles.append(int(row[2]))

    print(config.scale)
    tools = Tools()

    predictions = h5py.File(args.predictionfile, 'r')['predictions']

    tile_ind = 0
    for i in range (0, tiles.__len__()):
        if (tiles[i] == 0):
            images.append(Image.pre_process_img(imread(path_to_img[i]), color='gray'))
            scales_new.append(scales[i])
    # Create and save the reconstructed images
    reconstructed_predictions, reconstructed_masks = tools.reconstruct_images(images=images,predictions=predictions,scales=scales_new,rescale=config.scale,overlap=config.overlap,config=config)
    pickle.dump(({"masks":reconstructed_masks, "predictions":reconstructed_predictions}),open(os.path.join(os.path.dirname(args.predictionfile), os.path.basename(args.predictionfile).replace('.h5','_reconstructed.pkl')),"wb"))
main()