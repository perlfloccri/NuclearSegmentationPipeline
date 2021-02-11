#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../util.h"
#include "../matrix.h"
#include "../imageProcessing.h"
#include "sobelPrimitives.h"
#include "sobelTasks.h"
#include "primitiveOperations.h"
#include "cellCandidates.h"

extern MATRIX topSobels, bottomSobels, rightSobels, leftSobels;
extern writeSobels;
int writeSobelFlag;

extern MATRIX topRes, bottomRes, rightRes, leftRes;

/********************************************************************/
/********************************************************************/
/* It defines the Sobel primitives from scratch						*/
/* Previously defined primitives will be overwritten				*/
/********************************************************************/
/********************************************************************/
void findSobelPrimitivesFromScratch(CELL_IMAGE *im, int wthr, int d, int cid, MATRIX res){
    int j; char temp[100];
	markSobelPrimitives(im,cid,BOTTOM,wthr,d,res);
	markSobelPrimitives(im,cid,TOP,wthr,d,res);
	markSobelPrimitives(im,cid,LEFT,wthr,d,res);
	markSobelPrimitives(im,cid,RIGHT,wthr,d,res);
         
	findComponentBasedPrimitives(im,cid);

}
// This function marks the primitives with 1 and the background with 0 -- no primitive ids
void markSobelPrimitives(CELL_IMAGE *im, int cid, int stype, int wthr, int d, MATRIX res){
	MATRIX tmp	= allocateMatrix(im->dx,im->dy);
	int minx	= im->bnd.data[cid][MINX];
	int maxx	= im->bnd.data[cid][MAXX];
	int miny	= im->bnd.data[cid][MINY];
	int maxy	= im->bnd.data[cid][MAXY];
	int i, j;


	thresholdSobelValues(im,cid,stype);
	imfillSobelPrimitives(&(im->sobelPrims[stype]),minx,maxx,miny,maxy);
	
	findSobelBoundaries(im,stype,minx,maxx,miny,maxy,d);
	defineSobelPrimitives4ImageBoundaries(im,stype);
	eliminatePrimitivesBasedOnResults(im->resBnd[stype],&(im->sobelPrims[stype]),stype,res,minx,maxx,miny,maxy,d,tmp);
	eliminateSmallSobelPrimitives(im,stype,minx,maxx,miny,maxy,wthr);


	freeMatrix(tmp);
}
/********************************************************************/
/********************************************************************/
/********************************************************************/
/********************************************************************/
int connectBottomRight(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy){
	int n1, n2, i, p1, p2, cno = 0;
	double cx = 0.0, cy = 0.0, avgR;

	n1 = getLeftOuterPixels(bndx1,bndy1,im->primCell[LEFT],*minx,*maxx,*miny,*maxy);
	n2 = getTopOuterPixels(bndx2,bndy2,im->primCell[TOP],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	n1 = getRightOuterPixels(bndx1,bndy1,im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);
	n2 = getBottomOuterPixels(bndx2,bndy2,im->primCell[BOTTOM],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	if ((n1 == 0) || (n2 == 0))
		return 0;

	avgR = computeCentroidRadius(im->aCell,*minx,*maxx,*miny,*maxy,&cx,&cy,&cno);

	setCell(&(im->aCell),*minx,*maxx,*miny,*maxy);
	return 1;
}
int connectTopRight(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy){
	int n1, n2, i, p1, p2, cno = 0;
	double cx = 0.0, cy = 0.0, avgR;

	n1 = getLeftOuterPixels(bndx1,bndy1,im->primCell[LEFT],*minx,*maxx,*miny,*maxy);
	n2 = getBottomOuterPixels(bndx2,bndy2,im->primCell[BOTTOM],*minx,*maxx,*miny,*maxy); //////
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	n1 = getRightOuterPixels(bndx1,bndy1,im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);
	n2 = getTopOuterPixels(bndx2,bndy2,im->primCell[TOP],*minx,*maxx,*miny,*maxy); ////////
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	if ((n1 == 0) || (n2 == 0))
		return 0;

	avgR = computeCentroidRadius(im->aCell,*minx,*maxx,*miny,*maxy,&cx,&cy,&cno);

	setCell(&(im->aCell),*minx,*maxx,*miny,*maxy);
	return 1;
}
int connectTopLeft(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy){
	int n1, n2, i, p1, p2, cno = 0;
	double cx = 0.0, cy = 0.0, avgR;

	n1 = getRightOuterPixels(bndx1,bndy1,im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);////
	n2 = getBottomOuterPixels(bndx2,bndy2,im->primCell[BOTTOM],*minx,*maxx,*miny,*maxy); //////
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	n1 = getLeftOuterPixels(bndx1,bndy1,im->primCell[LEFT],*minx,*maxx,*miny,*maxy);
	n2 = getTopOuterPixels(bndx2,bndy2,im->primCell[TOP],*minx,*maxx,*miny,*maxy); ////////
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	if ((n1 == 0) || (n2 == 0))
		return 0;

	avgR = computeCentroidRadius(im->aCell,*minx,*maxx,*miny,*maxy,&cx,&cy,&cno);

	setCell(&(im->aCell),*minx,*maxx,*miny,*maxy);
	return 1;
}
int connectBottomLeft(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy){
	int n1, n2, i, p1, p2, cno = 0;
	double cx = 0.0, cy = 0.0, avgR;

	n1 = getRightOuterPixels(bndx1,bndy1,im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);
	n2 = getTopOuterPixels(bndx2,bndy2,im->primCell[TOP],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	n1 = getLeftOuterPixels(bndx1,bndy1,im->primCell[LEFT],*minx,*maxx,*miny,*maxy);
	n2 = getBottomOuterPixels(bndx2,bndy2,im->primCell[BOTTOM],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	if ((n1 == 0) || (n2 == 0))
		return 0;

	avgR = computeCentroidRadius(im->aCell,*minx,*maxx,*miny,*maxy,&cx,&cy,&cno);

	setCell(&(im->aCell),*minx,*maxx,*miny,*maxy);
	return 1;
}
int connectLeftRight(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2,
					  int *minx, int *maxx, int *miny, int *maxy, int flag){ // flag olmayan tip
	int n1, n2, i, p1, p2, cno = 0;
	double cx = 0.0, cy = 0.0, avgR;

	if (flag == TOP)		n2 = getBottomOuterPixels(bndx2,bndy2,im->primCell[BOTTOM],*minx,*maxx,*miny,*maxy);
	else					n2 = getTopOuterPixels(bndx2,bndy2,im->primCell[TOP],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	n1 = getLeftOuterPixels(bndx1,bndy1,im->primCell[LEFT],*minx,*maxx,*miny,*maxy);
	n2 = getRightOuterPixels(bndx2,bndy2,im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	if ((n1 == 0) || (n2 == 0))
		return 0;

	avgR = computeCentroidRadius(im->aCell,*minx,*maxx,*miny,*maxy,&cx,&cy,&cno);


	setCell(&(im->aCell),*minx,*maxx,*miny,*maxy);
	return 1;
}
int connectTopBottom(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2,
					  int *minx, int *maxx, int *miny, int *maxy, int flag){ // flag olmayan tip
	int n1, n2, i, p1, p2, cno = 0;
	double cx = 0.0, cy = 0.0, avgR;

	if (flag == RIGHT)		n2 = getLeftOuterPixels(bndx2,bndy2,im->primCell[LEFT],*minx,*maxx,*miny,*maxy);
	else					n2 = getRightOuterPixels(bndx2,bndy2,im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	n1 = getTopOuterPixels(bndx1,bndy1,im->primCell[TOP],*minx,*maxx,*miny,*maxy);
	n2 = getBottomOuterPixels(bndx2,bndy2,im->primCell[BOTTOM],*minx,*maxx,*miny,*maxy);
	for (i = 0; i < n1; i++)	im->aCell.data[bndx1[i]][bndy1[i]] = 2;
	for (i = 0; i < n2; i++)	im->aCell.data[bndx2[i]][bndy2[i]] = 2;

	if ((n1 == 0) || (n2 == 0))
		return 0;

	avgR = computeCentroidRadius(im->aCell,*minx,*maxx,*miny,*maxy,&cx,&cy,&cno);

	setCell(&(im->aCell),*minx,*maxx,*miny,*maxy);

	return 1;
}
int getSingleCell(CELL_IMAGE *im, int *minx, int *maxx, int *miny, int *maxy, int cells[4], int inc,
				   int *bndx1, int *bndx2, int *bndy1, int *bndy2, MATRIX tmp){
	int ng[4][4] = {0}, res;

	ng[TOP][RIGHT]		= ng[RIGHT][TOP]	= arePrimCellsNeighbor(im->primCell[TOP],im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);
	ng[TOP][LEFT]		= ng[LEFT][TOP]		= arePrimCellsNeighbor(im->primCell[TOP],im->primCell[LEFT],*minx,*maxx,*miny,*maxy);
	ng[BOTTOM][RIGHT]	= ng[RIGHT][BOTTOM] = arePrimCellsNeighbor(im->primCell[BOTTOM],im->primCell[RIGHT],*minx,*maxx,*miny,*maxy);
	ng[BOTTOM][LEFT]	= ng[LEFT][BOTTOM]	= arePrimCellsNeighbor(im->primCell[BOTTOM],im->primCell[LEFT],*minx,*maxx,*miny,*maxy);

	if ((ng[RIGHT][BOTTOM] == 0) && (ng[LEFT][BOTTOM] == 0))	res = connectLeftRight(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy,BOTTOM);
	else if ((ng[RIGHT][TOP] == 0) && (ng[LEFT][TOP] == 0))		res = connectLeftRight(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy,TOP);
	else if ((ng[RIGHT][TOP] == 0) && (ng[RIGHT][BOTTOM] == 0))	res = connectTopBottom(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy,RIGHT);
	else if ((ng[LEFT][TOP] == 0) && (ng[LEFT][BOTTOM] == 0))	res = connectTopBottom(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy,LEFT);
	else if (ng[TOP][RIGHT] == 0)								res = connectTopRight(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy);
	else if (ng[BOTTOM][RIGHT] == 0)							res = connectBottomRight(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy);
	else if (ng[TOP][LEFT] == 0)								res = connectTopLeft(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy);
	else if (ng[BOTTOM][LEFT] == 0)								res = connectBottomLeft(im,inc,bndx1,bndy1,bndx2,bndy2,minx,maxx,miny,maxy);

	if (res == 0)
		return 0;
	return 1;
}
void takePrimitives(CELL_IMAGE *im, int cells[4], int *rightBnd, int *leftBnd, int *topBnd, int *bottomBnd,
					int *minx, int *maxx, int *miny, int *maxy, MATRIX tmp){
	int i, j, k;

	findCandidateCellBoundingBox(im->prims,cells,minx,maxx,miny,maxy);
	findRightBoundaryPixels(im->sobelPrims[RIGHT],cells[RIGHT],*minx,*maxx,*miny,*maxy,rightBnd);
	findLeftBoundaryPixels(im->sobelPrims[LEFT],cells[LEFT],*minx,*maxx,*miny,*maxy,leftBnd);
	findTopBoundaryPixels(im->sobelPrims[TOP],cells[TOP],*minx,*maxx,*miny,*maxy,topBnd);
	findBottomBoundaryPixels(im->sobelPrims[BOTTOM],cells[BOTTOM],*minx,*maxx,*miny,*maxy,bottomBnd);
    
	initializeMatrixPartial(&(im->aCell),0,*minx,*maxx,*miny,*maxy);
	for (k = 0; k < 4; k++){
		initializeMatrixPartial(&(im->primCell[k]),0,*minx,*maxx,*miny,*maxy);
		if (cells[k] == -1)
			continue;
		takePrimitivesWithinBoundaries(im->sobelPrims[k],cells[k],*minx,*maxx,*miny,*maxy,rightBnd,leftBnd,
									   bottomBnd,topBnd,&(im->primCell[k]),tmp);
		for (i = *minx; i <= *maxx; i++) //primCell ile cell doldur...
			for (j = *miny; j <= *maxy; j++)
				if (im->primCell[k].data[i][j])
					im->aCell.data[i][j] = 1;
	}
	
}
int findSingleCell(CELL_IMAGE *im, int ids[4], int inc, int satisyfyFlag, int smallWidthThr, int primCnt, int *minx, int *miny,
				   int *maxx, int *maxy,int *bndx1, int *bndx2, int *bndy1, int *bndy2, MATRIX tmp, MATRIX tmp2){
	int res;

	takePrimitives(im,ids,bndy1,bndy2,bndx1,bndx2,minx,maxx,miny,maxy,tmp);

    if (satisyfyFlag == YES){
		res = satisfyWidthConditions(*im,*minx,*maxx,*miny,*maxy,ids,smallWidthThr,primCnt);
		if (!res)
			return 0;
	}
	if (getSingleCell(im,minx,maxx,miny,maxy,ids,inc,bndx1,bndx2,bndy1,bndy2,tmp))
		return 1;
	return 0;
}
void findValidNeighbors(CELL_IMAGE *im, int cid, int smallWidthThr, CELLS *cand, int inc, int gtype){
	int *bndy1		= (int *)calloc((im->dx)*(im->dy),sizeof(int));
	int *bndy2		= (int *)calloc((im->dx)*(im->dy),sizeof(int));
	int *bndx1		= (int *)calloc((im->dy)*(im->dx),sizeof(int));
	int *bndx2		= (int *)calloc((im->dy)*(im->dx),sizeof(int));
	MATRIX tmp		= allocateMatrix(im->dx,im->dy);
	MATRIX tmp2		= allocateMatrix(im->dx,im->dy);
	int cells[MAXCELL][4], cnt = 0, i, j;
	int minx, maxx, miny, maxy, res;
    char temp[100];
    int counter = 0;
	findSobelNeighbors(im,im->bnd.data[cid][MINX],im->bnd.data[cid][MAXX],im->bnd.data[cid][MINY],im->bnd.data[cid][MAXY]);

	if (gtype == FOUR){
		find4Components(im->adj,im->prims,im->primNo,RIGHT,BOTTOM,LEFT,TOP,cells,&cnt);
		find4Components(im->adj,im->prims,im->primNo,RIGHT,TOP,LEFT,BOTTOM,cells,&cnt);
		find4Components(im->adj,im->prims,im->primNo,LEFT,BOTTOM,RIGHT,TOP,cells,&cnt);
		find4Components(im->adj,im->prims,im->primNo,LEFT,TOP,RIGHT,BOTTOM,cells,&cnt);
	}
	else{
		find3Components(im->adj,im->prims,im->primNo,RIGHT,BOTTOM,LEFT,TOP,cells,&cnt);
		find3Components(im->adj,im->prims,im->primNo,RIGHT,TOP,LEFT,BOTTOM,cells,&cnt);
		find3Components(im->adj,im->prims,im->primNo,BOTTOM,RIGHT,TOP,LEFT,cells,&cnt);
		find3Components(im->adj,im->prims,im->primNo,BOTTOM,LEFT,TOP,RIGHT,cells,&cnt);
	}

	cand->cellNo = 0;

	for (i = 0; i < cnt; i++){       
        res = findSingleCell(im,cells[i],inc,YES,smallWidthThr,gtype,&minx,&miny,&maxx,&maxy,bndx1,bndx2,bndy1,bndy2,tmp,tmp2);
		if (res){
			add2CellCandidates(cand,im->aCell,cells[i],minx,maxx,miny,maxy,gtype);             
        }
	}

	free(bndx1);
	free(bndx2);
	free(bndy1);
	free(bndy2);
	freeMatrix(tmp);
	freeMatrix(tmp2);
}
void updateResultMatrix(CELL_IMAGE *im, int ids[4], MATRIX *res, int resLabel, int minx, int maxx, int miny, int maxy){
	int i, j;

	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if ((res->data[i][j] == 0) && im->aCell.data[i][j]){
				res->data[i][j] = resLabel;
				if (im->sobelPrims[RIGHT].data[i][j] == ids[RIGHT])			im->resBnd[RIGHT].data[i][j] = 1;
				if (im->sobelPrims[LEFT].data[i][j] == ids[LEFT])			im->resBnd[LEFT].data[i][j] = 1;
				if (im->sobelPrims[TOP].data[i][j] == ids[TOP])				im->resBnd[TOP].data[i][j] = 1;
				if (im->sobelPrims[BOTTOM].data[i][j] == ids[BOTTOM])		im->resBnd[BOTTOM].data[i][j] = 1;
			}
}
void updateResultSobels(CELL_IMAGE *im, int ids[4], MATRIX res, int minx, int maxx, int miny, int maxy){
	int i, j;

	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if ((res.data[i][j] == 0)	&& im->aCell.data[i][j]){
				if ((ids[RIGHT] != -1)	&& (im->sobelPrims[RIGHT].data[i][j] == ids[RIGHT]))		rightRes.data[i][j] = 1;
				if ((ids[LEFT] != -1)	&& (im->sobelPrims[LEFT].data[i][j] == ids[LEFT]))			leftRes.data[i][j] = 1;
				if ((ids[TOP] != -1)	&& (im->sobelPrims[TOP].data[i][j] == ids[TOP]))			topRes.data[i][j] = 1;
				if ((ids[BOTTOM] != -1) && (im->sobelPrims[BOTTOM].data[i][j] == ids[BOTTOM]))		bottomRes.data[i][j] = 1;
			}
}
void takeCellAndUpdate(CELL_IMAGE *im, int cid, CELLS *C, int cellId, MATRIX *res, int resLabel, int wthr, int inc,
					   int *bndx1, int *bndy1, int *bndx2, int *bndy2, MATRIX tmp, MATRIX tmp2, int gtype){
	int minx, maxx, miny, maxy;

	findSingleCell(im,C->cells[cellId].ids,inc,NO,0,0,&minx,&miny,&maxx,&maxy,bndx1,bndx2,bndy1,bndy2,tmp,tmp2);

	if (writeSobels == 2)
		updateResultSobels(im,C->cells[cellId].ids,*res,minx,maxx,miny,maxy);
	updateResultMatrix(im,C->cells[cellId].ids,res,resLabel,minx,maxx,miny,maxy);

	C->cells[cellId].taken = 1;

	if (C->cells[cellId].ids[TOP] != -1)			updateSobelPrimitives(im,C->cells[cellId].ids[TOP],wthr,tmp);
	if (C->cells[cellId].ids[RIGHT] != -1)			updateSobelPrimitives(im,C->cells[cellId].ids[RIGHT],wthr,tmp);
	if (C->cells[cellId].ids[LEFT] != -1)			updateSobelPrimitives(im,C->cells[cellId].ids[LEFT],wthr,tmp);
	if (C->cells[cellId].ids[BOTTOM] != -1)			updateSobelPrimitives(im,C->cells[cellId].ids[BOTTOM],wthr,tmp);

	findValidNeighbors(im,cid,wthr,C,inc,gtype);
}
int selectBestCell(CELLS C, int critType, double *crit){
	int selCell = -1, i;
	double bestValue = 0;

	for (i = 0; i < C.cellNo; i++){

		if (C.cells[i].taken)
			continue;

		if (selCell == -1){
			selCell = i;
			switch (critType) {
				case STDR:
					bestValue = C.cells[i].stdR;
					break;
				case STDR_AVGR:
					bestValue = C.cells[i].stdR / C.cells[i].avgR;
					break;
			}
			continue;
		}
		switch (critType) {
			case STDR:
				if (C.cells[i].stdR < bestValue){
					bestValue = C.cells[i].stdR;
					selCell = i;
				}
				break;
			case STDR_AVGR:
				if (C.cells[i].stdR / C.cells[i].avgR < bestValue){
					bestValue = C.cells[i].stdR / C.cells[i].avgR;
					selCell = i;
				}
				break;
		}
	}
	*crit = bestValue;
	return selCell;
}
void updateSobels(MATRIX *res, MATRIX conn, int cid, MATRIX prims){
	int i, j;

	for (i = 0; i < conn.row; i++)
		for (j = 0; j < conn.column; j++)
			if ((conn.data[i][j] == cid) && (prims.data[i][j] >= 0))
				res->data[i][j] = 1;
}
void processComponents(CELL_IMAGE *im, int cid, int wthr, int d1, int d2, MATRIX *res, double otsuPerc,
					   int critType, int critProp, double critInitThr, int inc){
    char temp[100];
	int *bndy1		= (int *)calloc((im->dx)*(im->dy),sizeof(int));
	int *bndy2		= (int *)calloc((im->dx)*(im->dy),sizeof(int));
	int *bndx1		= (int *)calloc((im->dy)*(im->dx),sizeof(int));
	int *bndx2		= (int *)calloc((im->dy)*(im->dx),sizeof(int));
	MATRIX tmp		= allocateMatrix(im->dx,im->dy);
	MATRIX tmp2		= allocateMatrix(im->dx,im->dy);
	CELLS C			= allocateCellCandidates(1000,im->dx,im->dy);
	int cellLabel	= maxMatrixEntry(*res) + 1;
	double currPerc = 1.0, crit;
	int i, j, selCell, count = 0, iter = 1;

	initialize4ConnectedComponent(im,cid);
	while (currPerc >= otsuPerc){
  
		findSobelPrimitivesFromScratch(im,wthr,d1,cid,*res);
        
		if ((writeSobels == 1) && (writeSobelFlag == 1)){
			updateSobels(&topSobels,im->conn,cid,im->sobelPrims[TOP]);
			updateSobels(&bottomSobels,im->conn,cid,im->sobelPrims[BOTTOM]);
			updateSobels(&leftSobels,im->conn,cid,im->sobelPrims[LEFT]);
			updateSobels(&rightSobels,im->conn,cid,im->sobelPrims[RIGHT]);
			writeSobelFlag = 2;			
		}
                                               
		// --------------------------------------------------
		// First FOUR primitives
		// --------------------------------------------------
		findValidNeighbors(im,cid,wthr,&C,inc,FOUR);
		if (im->primNo <= 1)
			break;        
                 
		while (1){	
			selCell = selectBestCell(C,critType,&crit);
			if (selCell == -1)
				break;
			if ((critProp == MAXCRIT) && (crit < critInitThr))
				break;
			if ((critProp == MINCRIT) && (crit > critInitThr))
				break;

			takeCellAndUpdate(im,cid,&C,selCell,res,cellLabel,wthr,inc,bndx1,bndy1,bndx2,bndy2,tmp,tmp2,FOUR);
			    
			cellLabel++;
		}
        
		// --------------------------------------------------
		// Then THREE primitives
		// --------------------------------------------------
		findValidNeighbors(im,cid,wthr,&C,inc,THREE);
		if (im->primNo <= 1)
			break;
   
		while (1){
			selCell = selectBestCell(C,critType,&crit);
			if (selCell == -1)
				break;
			if ((critProp == MAXCRIT) && (crit < critInitThr))
				break;
			if ((critProp == MINCRIT) && (crit > critInitThr))
				break;

			takeCellAndUpdate(im,cid,&C,selCell,res,cellLabel,wthr,inc,bndx1,bndy1,bndx2,bndy2,tmp,tmp2,THREE);  
			cellLabel++;
		}
        
		if (critProp == MAXCRIT)
			critInitThr *= 0.9;
		if (critProp == MINCRIT)
			critInitThr *= 1.1;

		currPerc *= 0.9;
		for (i = 0; i < 4; i++)
			im->thr.data[cid][i] *= 0.9;
   
	}
	free(bndx1);
	free(bndy1);
	free(bndx2);
	free(bndy2);
	freeMatrix(tmp);
	freeMatrix(tmp2);
	freeCellCandidates(C);
}
void processIndividualConnectedComponent(CELL_IMAGE *im, int cid, MATRIX *res, int wthr, int d1, int d2, int critType,
										 double otsuPercMin, double critInitThr){
	double initThr[4];
	int i;
	int critProp;
	int inc = 3;

	if (critType == STDR)
		critProp = MINCRIT;
	else if (critType == STDR_AVGR)
		critProp = MINCRIT;

	for (i = 0; i < 4; i++)
		initThr[i] = im->thr.data[cid][i];

	writeSobelFlag = 1;
	processComponents(im,cid,wthr,d1,d2,res,otsuPercMin,critType,critProp,critInitThr,inc);
  
	printf("%d out of %d\n",cid,im->connNo);
}
