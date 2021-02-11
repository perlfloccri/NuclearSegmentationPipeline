#ifndef __watershed_h
#define __watershed_h

#define MAXHEAPSIZE		1000000

#include "../matrix.h"
#include "../util.h"
#include "../imageProcessing.h"

void readPrims(MATRIX prims[4], char *fname);
void freePrims(MATRIX prims[4]);
MATRIX markCellCentroid(MATRIX primRes);

// -------------------------------------------------------------------------------------------------- //
// ------------ TO FIND THE OUTER BOUNDARIES OF THE PRIMITIVES -------------------------------------- //
// -------------------------------------------------------------------------------------------------- //
void markLeftOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndy,
							 int *firstX, int *firstY, int *lastX, int *lastY);
void markRightOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndy,
							  int *firstX, int *firstY, int *lastX, int *lastY);
void markTopOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndx,
							int *firstX, int *firstY, int *lastX, int *lastY);
void markBottomOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndx,
							   int *firstX, int *firstY, int *lastX, int *lastY);

int arePrimNeighbors(MATRIX primRes, int cid, MATRIX p1, MATRIX p2, BNDBOX bnd);
int isEightAdjacent(int x, int y, MATRIX M);
void connectTopLeftOuter(MATRIX *outer, int topX, int topY, int lefX, int lefY);
void connectTopRightOuter(MATRIX *outer, int topX, int topY, int rigX, int rigY);
void connectBottomLeftOuter(MATRIX *outer, int botX, int botY, int lefX, int lefY);
void connectBottomRightOuter(MATRIX *outer, int botX, int botY, int rigX, int rigY);

MATRIX findOuterBackground(MATRIX primRes, MATRIX prims[4], int d);
void growOuterBoundaries(MATRIX *outer, MATRIX primRes, MATRIX prims[4], int d);


// -------------------------------------------------------------------------------------------------- //
// ----------------- MODIFIED FLOODING OF THE WATERSHED ALGORITHM ----------------------------------- //
// -------------------------------------------------------------------------------------------------- //
void takeLastPrimMarkers(MATRIX *markers, MATRIX floodTimer, int *lastPrim);
int initializeWatershed(MATRIX *markers, MATRIX *floodTimer, int **x, int **y, int **lastPrim, HEAP *h);
void watershedPrimUpdate(MATRIX *markers, MATRIX primRes, MATRIX mask, MATRIX outer, int pathDist);
void watershedPrimUpdateWithoutMask(MATRIX *markers, MATRIX primRes, MATRIX outer, int pathDist);


void postProcess(MATRIX *markers, int radius);
int checkRegionGrowingArguments(int argc, char *argv[]);
void findCellBoundaries(MATRIX M);
void REGION_GROWING(int argc, char *argv[]);


#endif
