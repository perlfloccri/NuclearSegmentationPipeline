%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%   Iterative H-minima Based Marker-Controlled Watershed for Cell Nucleus
%   Segmentation 
%
%   The proposed algorithm iteratively identifies markers, considering 
%   a set of different h values. In	each iteration, it defines a set of 
%   candidates using a particular h value and selects the markers from 
%   those candidates provided that they fulfill the size requirement.
%   
%   NOTE: The following source codes and executable are provided for 
%   research purposes only. The authors have no responsibility for any 
%   consequences of use of these source codes and the executable. If you 
%   use any part of these codes, please cite the following paper.
%   
%   C. F. Koyuncu, E. Akhan, T. Ersahin, R. Cetin-Atalay, C. Gunduz-Demir,
%   "Iterative h-minima based marker-controlled watershed for cell nucleus
%   segmentation, under revision.
%
%
%   If that is YOUR FIRST USE of this program
%   uncomment the line 65, and call compileCCodes function
%   This function produces a matlab executable called "flooding"
%
%
%   This function takes five inputs and outputs the segmentation results
%   inputName:  Image filename (can be an RGB or a grayscale image)
%   tArea:      The area threshold
%   dSize:      The size of the disk structuring elements and the
%               average filter
%   gAngle:     The	angle to define the start and end points of an arc,
%               whose pixels are used to define the stopping condition of
%               the flooding process
%   offset:     The offset is the maximum number of pixels that a marker
%               grows at the end without considering the stopping condition
%   segmRes:    The segmentation map. Each cell nucleus is labeled with
%               a different integer, labels are from 1 to N. Background is
%               labeled with 0.
%
%   Example use: 
%               results = iterativeHmin('huh7f_4.jpg', 20, 10, 15, 2);
% 
%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function segmRes = iterativeHmin(inputName, tArea, dSize, gAngle, offset)
    % Read the input image and convert it to the grayscale if necessary
    grayIm  = readInputImage(inputName);

    % 1. Map construction
    binMask = im2bw(grayIm, graythresh(grayIm) * 0.5);
    gMap    = obtainMap(grayIm, dSize);

	% 2. Iterative marker identification
    initH   = 1;
    [markers, markersOrg] = findMarkers(gMap, binMask, tArea, dSize, initH);

    % 3. Region growing
    % You need run the following code and compile C codes by the mex 
    % compiler by calling the following function
    % the name of the executable should be flooding
    
    % compileCCodes % call this function just once
    segmRes = flooding(double(markers), double(binMask), gAngle, offset);
end
%------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function compileCCodes

cd floodingCodes
mex ./matrix.c -c
mex ./util.c -c
mex util.obj matrix.obj watershed.c -output ../flooding
cd ../

end
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function grayIm = readInputImage(inputName)
    
    grayIm = imread (inputName);
    if size(grayIm,3) == 3
        grayIm = rgb2gray(grayIm);
    end
    
end    
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function gMap = obtainMap(grayIm, dSize)

    grayIm = imopen(grayIm, strel('disk', dSize));
    gMap   = sobelFilter(grayIm);
    gMap   = imfilter(gMap, fspecial('average', 2 * dSize - 1));    
end
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function res = sobelFilter(im)

    im   = double(im);    
    opX  = fspecial('sobel');
    opY  = opX';
    xDer = imfilter(im, opX, 'replicate');
    yDer = imfilter(im, opY, 'replicate');
    res  = sqrt(xDer .* xDer + yDer .* yDer);

end
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%