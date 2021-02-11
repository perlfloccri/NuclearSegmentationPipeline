# Iterative H-minima Based Marker-Controlled Watershed for Cell Nucleus Segmentation
The proposed algorithm iteratively identifies markers, considering a set of different h values. In each iteration, it defines a set of candidates using a particular h value and selects the markers from those candidates provided that they fulfill the size requirement.

**NOTE: The following source codes are provided for research purposes only. The authors have no responsibility for any consequences of use of these source codes. If you use any part of these codes, please cite the following paper.**

>C. F. Koyuncu, E. Akhan, T. Ersahin, R. Cetin-Atalay, C. Gunduz-Demir, "Iterative h-minima based marker-controlled watershed for cell nucleus segmentation", Cytometry: Part A, 89A:338-349, 2016.

Please contant me for further questions at canfkoyuncu@gmail.com.

### Source code
You need to call the `iterativeHmin` function in Matlab. This function uses Matlab codes as well as C codes, which should first be compiled by the mex compiler in Matlab.

For the first use of this program, please uncomment line 65 (line including compileCCodes statement) in the iterativeHmin.m file. This line calls `compileCCodes` function, which compiles the C codes by the mex compiler and creates an executable called *flooding* that will be used by the `iterativeHmin` function. After your first use, you may comment this line.

The `iterativeHmin` function has the following prototype:

**`function segmRes = iterativeHmin (inputName, tArea, dSize, gAngle, offset)`**

in which
* ___inputName___	:	The name of the image file. It can be an RGB or a grayscale image.
* ___tArea___	:	The area threshold.
* ___dSize___ :	The size of the disk structuring elements and the average filter.
* ___gAngle___ :	The	angle to define the start and end points of an arc, whose pixels are used to define the stopping condition of the flooding process.
* ___offset___ :	The offset is the maximum number of pixels that a marker grows at the end without considering the stopping condition.
* ___segmRes___ :	The output segmentation map. Pixels of the same cell nucleus are labeled with the same integer, where labels are from 1 to N. Background pixels are labeled with 0.

Example use:
`results = iterativeHmin('huh7f_4.jpg', 20, 10, 15, 2);`

Example output:

<img src="./huh7f_4.jpg" title="Example fluorescence image" width=400 /> <img src="./huh7f_4_res.png" title="Segmentation results (green boundaries) together with the actual cell centroids (red dots)" width=400 />
