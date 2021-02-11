#ifndef __sobelPrimitives_h
#define __sobelPrimitives_h

#define MAXCELL 20000

#include "../matrix.h"
#include "cellCandidates.h"


#define STDR		1
#define STDR_AVGR	2
#define ROUNDNESS	3

#define MAXCRIT		1
#define MINCRIT		2

struct TPrimitive{
	int ptype;
	int minx, maxx, miny, maxy;
	double cx, cy;
	int area;
	double sbAvg;
};
typedef struct TPrimitive PRIMITIVE;

struct TCellImage{
	MATRIX sobelValues[4];	// sobel filter values
	MATRIX sobelPrims[4];	// sobel primitive maps -- background should be -1
	MATRIX conn;			// background should be -1
	int connNo;				// maxMatrixEntry(conn)
	int dx, dy;
	MATRIX bnd;
	MATRIX thr;
	
	int primNo;
	int maxNo;
	PRIMITIVE *prims;
	MATRIX mask;
	MATRIX adj;
	MATRIX aCell;
	MATRIX primCell[4];
	MATRIX resBnd[4];
};
typedef struct TCellImage CELL_IMAGE;

///******************************************************************/
///******************************************************************/
///******************************************************************/
void findCellImageConnectedComponents(CELL_IMAGE *im, MATRIX mask);
void computeSobelValues(CELL_IMAGE *im, MATRIX gray);
void computeComponentSobelThresholds(CELL_IMAGE *im, int stype);

CELL_IMAGE createCellImage(MATRIX gray, MATRIX mask);
void freeCellImage(CELL_IMAGE im);
///******************************************************************/
///******************************************************************/
///******************************************************************/
void thresholdSobelValues(CELL_IMAGE *im, int cid, int stype);
void eliminateSmallSobelPrimitives(CELL_IMAGE *im, int stype, int minx, int maxx, int miny, int maxy, int wthr);
void findSobelBoundaries(CELL_IMAGE *im, int stype, int minx, int maxx, int miny, int maxy, int d);
void defineSobelPrimitives4ImageBoundaries(CELL_IMAGE *im, int stype);
int componentBasedEightConnectivity(MATRIX *sb, int minx, int maxx, int miny, int maxy, int nStart, MATRIX res);
void findComponentBasedPrimitives(CELL_IMAGE *im, int cid);
void extractProperties4AllPrimitives(CELL_IMAGE *im, int stype, int minx, int maxx, int miny, int maxy, int startNo, int endNo);
///******************************************************************/
///******************************************************************/
///******************************************************************/
void findNeighborPixels(MATRIX *adj, int **r1, int **r2, int minx, int maxx, int miny, int maxy);
void findSobelNeighbors(CELL_IMAGE *im, int minx, int maxx, int miny, int maxy);	
void find4Components(MATRIX adj, PRIMITIVE *curr, int primNo, int s1, int s2, int s3, int s4, int cells[MAXCELL][4], int *cnt);
void find3Components(MATRIX adj, PRIMITIVE *curr, int primNo, int s1, int s2, int s3, int s4, int cells[MAXCELL][4], int *cnt);
void findCandidateCellBoundingBox(PRIMITIVE *curr, int cells[4], int *minx, int *maxx, int *miny, int *maxy);
int satisfyWidthConditions(CELL_IMAGE im, int minx, int maxx, int miny, int maxy, int *cellIds, int smallThr, int total);
///******************************************************************/
///******************************************************************/
///******************************************************************/
void initialize4ConnectedComponent(CELL_IMAGE *im, int cid);
void extendPrimitivesSize(CELL_IMAGE *im, int newSize);
void updateSobelPrimitives(CELL_IMAGE *im, int pid, int wthr, MATRIX tmp);

#endif


