function [ mask ] = ARG_initial_segmentation(im, openSize, afterSize, areaThr, gthr)
%ARG_INITIAL_SEGMENTATION Creates a binary mask for a fluorescence cellular image
%   Creates a binary mask that can be used as (copy from paper)
%
% Citation: S. Arslan, T. Ersahin, R. Cetin-Atalay, and C. Gunduz-Demir, 
% "Attributed relational graphs for cell nucleus segmentation in fluorescence 
% microscopy images," IEEE Transactions on Medical Imaging, vol. 32, no. 6, 
% pp. 1121-1131, 2013. 
%
% author: Salim Arslan (name.surname@imperial.ac.uk)
% last updated: 21.06.2014
%
% input
%   im: 3D RGB image
%   openSize: Size of structural element for opening
%   afterSize: Size of structural element for opening after binarization
%   areaThr: Threshold for clearing noise from the mask
%   gthr: Otsu threshold level weight
%
% output
%   mask: Binary mask
%
% Parameter values used in the paper experiments:
% gthr = 0.5;
% areaThr = 250;
% openSize = 50;
% afterSize = 5;

gray_im = im;%im(:,:,3); 
se = strel('disk', openSize);
opened_im = imopen(gray_im, se);
gray_im = gray_im - opened_im;

binarized = im2bw(gray_im, graythresh(gray_im) * gthr);

se = strel('disk', afterSize);
binarized = imopen(binarized, se);
binarized = bwareaopen(binarized, areaThr);
mask = imdilate(binarized,se);

end

