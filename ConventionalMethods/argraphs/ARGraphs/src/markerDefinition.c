#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../matrix.h"
#include "../util.h"
#include "../imageProcessing.h"
#include "sobelPrimitives.h"
#include "sobelTasks.h"
#include "cellCandidates.h"

MATRIX topSobels, bottomSobels, rightSobels, leftSobels;
int writeSobels;

MATRIX topRes, bottomRes, rightRes, leftRes;

void allocateGlobalSobels(int dx, int dy){
	topSobels = allocateMatrix(dx,dy);
	bottomSobels = allocateMatrix(dx,dy);
	rightSobels = allocateMatrix(dx,dy);
	leftSobels = allocateMatrix(dx,dy);

	initializeMatrix(&topSobels,0);
	initializeMatrix(&bottomSobels,0);
	initializeMatrix(&rightSobels,0);
	initializeMatrix(&leftSobels,0);
}

void allocateGlobalResults(int dx, int dy){
	topRes = allocateMatrix(dx,dy);
	bottomRes = allocateMatrix(dx,dy);
	rightRes = allocateMatrix(dx,dy);
	leftRes = allocateMatrix(dx,dy);

	initializeMatrix(&topRes,0);
	initializeMatrix(&bottomRes,0);
	initializeMatrix(&rightRes,0);
	initializeMatrix(&leftRes,0);

}
void freeGlobalSobels(){
	freeMatrix(topSobels);
	freeMatrix(bottomSobels);
	freeMatrix(leftSobels);
	freeMatrix(rightSobels);
}
void freeGlobalResults(){
	freeMatrix(topRes);
	freeMatrix(bottomRes);
	freeMatrix(leftRes);
	freeMatrix(rightRes);
}
void writeGlobalSobels(char *fname){
	char str[1000];
    
	sprintf(str,"%s_left",fname);
	writeMatrixIntoFile(leftSobels,str,1);

	sprintf(str,"%s_right",fname);
	writeMatrixIntoFile(rightSobels,str,1);

	sprintf(str,"%s_top",fname);
	writeMatrixIntoFile(topSobels,str,1);

	sprintf(str,"%s_bottom",fname);
	writeMatrixIntoFile(bottomSobels,str,1);
}
void writeGlobalResults(char *fname){
	char str[1000];

	sprintf(str,"%s_left_res",fname);
	writeMatrixIntoFile(leftRes,str,1);

	sprintf(str,"%s_right_res",fname);
	writeMatrixIntoFile(rightRes,str,1);

	sprintf(str,"%s_top_res",fname);
	writeMatrixIntoFile(topRes,str,1);

	sprintf(str,"%s_bottom_res",fname);
	writeMatrixIntoFile(bottomRes,str,1);
}
void parseArguments(int argc, char *argv[], int *smallThr, int *critType, double *otsuPercMin, double *critInitThr, int *cid){

	if (argc != 10){
		printf("\nUsage:");
		printf("\n\t0.  Marker definition (1) or flooding (2)");
		printf("\n\t1.  Gray-blue filename [blue]");
		printf("\n\t2.  Initial segmentation filename [mask]");
		printf("\n\t3.  Result filename [prim]");
		printf("\n\t4.  T_size:  width-height threshold [15]");
		printf("\n\t5.  T_perc:  minimum Otsu percentage [0.3]");
		printf("\n\t6.  T_std:   initial criterion threshold [4.0]");
		printf("\n\t7.  Component id (-1 for all)");
		printf("\n\t8.  Create extra output? (0: NO, 1: YES--Initial Sobels, 2: YES--Result Sobels [essential for flooding])");
		printf("\n\n");
		exit(1);
	}

	*smallThr = atoi(argv[5]);
	if ((*smallThr) < 0)
		terminateProgram("Width-height threshold should be nonnegative");

	*critType = STDR;

	*otsuPercMin = atof(argv[6]);
	if ((*otsuPercMin <= 0) || (*otsuPercMin > 1))
		terminateProgram("Minimum Otsu percentage should be in between 0 and 1");

	*critInitThr = atof(argv[7]);
    
	*cid = atoi(argv[8]);

	writeSobels = atoi(argv[9]);

}
void writeBoundariesIntoFile(MATRIX res, char *fname){
	int k1, k2, tmp;
	MATRIX bnd = allocateMatrix(res.row,res.column);

	initializeMatrix(&bnd,0);
	for (k1 = 0; k1 < res.row; k1++)
		for (k2 = 0; k2 < res.column; k2++){
			if (res.data[k1][k2] == 0)
				continue;

			tmp = res.data[k1][k2];
			if ((k1 - 1 < 0) || (res.data[k1-1][k2] != tmp))
				bnd.data[k1][k2] = tmp;
			if ((k1 + 1 >= res.row) || (res.data[k1+1][k2] != tmp))
				bnd.data[k1][k2] = tmp;
			if ((k2 - 1 < 0) || (res.data[k1][k2-1] != tmp))
				bnd.data[k1][k2] = tmp;
			if ((k2 + 1 >= res.column) || (res.data[k1][k2+1] != tmp))
				bnd.data[k1][k2] = tmp;

		}
	writeMatrixIntoFile(bnd,fname,1);
	freeMatrix(bnd);
}

void MARKER_DEFINITION(int argc, char *argv[]){
	MATRIX gray, mask, res;
	CELL_IMAGE im;
	int d, cid, smallThr, critType;
	double otsuPercMin, critInitThr;
	char str[1000];

	parseArguments(argc,argv,&smallThr,&critType,&otsuPercMin,&critInitThr,&cid);

	gray	= readMatrix(argv[2]);
	mask	= readMatrix(argv[3]);
	im		= createCellImage(gray,mask);
	res		= allocateMatrix(im.dx,im.dy);
 
	initializeMatrix(&res,0);

	if (writeSobels == 1)
		allocateGlobalSobels(res.row,res.column);
	else if (writeSobels == 2)
		allocateGlobalResults(res.row,res.column);
     
	d = 3;
	if (cid == -1){
		for (cid = 0; cid < im.connNo; cid++)
			processIndividualConnectedComponent(&im,cid,&res,smallThr,d,d,critType,otsuPercMin,critInitThr);
	}
	else
		processIndividualConnectedComponent(&im,cid,&res,smallThr,d,d,critType,otsuPercMin,critInitThr);
 
   
	writeMatrixIntoFile(res,argv[4],1);

	if (writeSobels == 1){
        writeGlobalSobels(argv[2]);
		freeGlobalSobels();
	}
	else if (writeSobels == 2){
		writeGlobalResults(argv[4]);
		freeGlobalResults();
	}

	freeCellImage(im);
	freeMatrix(res);
	freeMatrix(gray);
	freeMatrix(mask);

}


