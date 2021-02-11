#ifndef __sobelTasks_h
#define __sobelTasks_h

#include "../matrix.h"
#include "cellCandidates.h"
#include "sobelPrimitives.h"

///******************************************************************/
///******************************************************************/
///******************************************************************/
void markSobelPrimitives(CELL_IMAGE *im, int cid, int stype, int wthr, int d, MATRIX res);
void findSobelPrimitivesFromScratch(CELL_IMAGE *im, int wthr, int d, int cid, MATRIX res);
///******************************************************************/
///******************************************************************/
///******************************************************************/
/********************************************************************/
/********************************************************************/
/********************************************************************/
int connectBottomRight(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy);
int connectTopRight(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy);
int connectTopLeft(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy);
int connectBottomLeft(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, int *minx, int *maxx, int *miny, int *maxy);
int connectLeftRight(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, 
					 int *minx, int *maxx, int *miny, int *maxy, int flag);
int connectTopBottom(CELL_IMAGE *im, int inc, int *bndx1, int *bndy1, int *bndx2, int *bndy2, 
					 int *minx, int *maxx, int *miny, int *maxy, int flag);
///******************************************************************/
///******************************************************************/
///******************************************************************/
int getSingleCell(CELL_IMAGE *im, int *minx, int *maxx, int *miny, int *maxy, int cells[4], int inc, 
				   int *bndx1, int *bndx2, int *bndy1, int *bndy2, MATRIX tmp);
int findSingleCell(CELL_IMAGE *im, int ids[4], int inc, int satisyfyFlag, int smallWidthThr, int primCnt, int *minx, int *miny, 
				   int *maxx, int *maxy,int *bndx1, int *bndx2, int *bndy1, int *bndy2, MATRIX tmp, MATRIX tmp2);
void takePrimitives(CELL_IMAGE *im, int cells[4], int *rightBnd, int *leftBnd, int *topBnd, int *bottomBnd, 
					int *minx, int *maxx, int *miny, int *maxy, MATRIX tmp);
void findValidNeighbors(CELL_IMAGE *im, int cid, int smallWidthThr, CELLS *cand, int inc, int gtype);
///******************************************************************/
///******************************************************************/
///******************************************************************/
void updateResultMatrix(CELL_IMAGE *im, int ids[4], MATRIX *res, int resLabel, int minx, int maxx, int miny, int maxy);
void updateResultSobels(CELL_IMAGE *im, int ids[4], MATRIX res, int minx, int maxx, int miny, int maxy);
void takeCellAndUpdate(CELL_IMAGE *im, int cid, CELLS *C, int cellId, MATRIX *res, int resLabel, int wthr, int inc,
			int *bndx1, int *bndy1, int *bndx2, int *bndy2, MATRIX tmp, MATRIX tmp2, int gtype);	
int selectBestCell(CELLS C, int critType, double *crit);
///******************************************************************/
///******************************************************************/
///******************************************************************/
void updateSobels(MATRIX *res, MATRIX conn, int cid, MATRIX prims);
void processComponents(CELL_IMAGE *im, int cid, int wthr, int d1, int d2, MATRIX *res, double otsuPerc,
		int critType, int critProp, double critInitThr, int inc);
void processIndividualConnectedComponent(CELL_IMAGE *im, int cid, MATRIX *res, int wthr, int d1, int d2, int critType, 
										 double otsuPercMin, double critInitThr);

#endif

