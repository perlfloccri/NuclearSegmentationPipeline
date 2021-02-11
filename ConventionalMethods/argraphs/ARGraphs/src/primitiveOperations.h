#ifndef __primitiveOperations_h
#define __primitiveOperations_h

#include "../matrix.h"

#define WIDTH 0
#define HEIGHT 1

#define LAST 0
#define FIRST 1

#define TOP_LABEL 1
#define BOTTOM_LABEL 2
#define RIGHT_LABEL 3
#define LEFT_LABEL 4

void imfillSobelPrimitives(MATRIX *sb, int minx, int maxx, int miny, int maxy);
void takeLargestComponent(MATRIX *M, int label, int minx, int maxx, int miny, int maxy, MATRIX res);
int connectedComponentsWithMask(MATRIX im, int cid, int minx, int maxx, int miny, int maxy, int startingLabel, int flag48, MATRIX *res);

void eliminateHorizontalSmallPrimitives(MATRIX *sb, int minx, int maxx, int miny, int maxy, int wthr);
void eliminateVerticalSmallPrimitives(MATRIX *sb, int minx, int maxx, int miny, int maxy, int wthr);

void findRightPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d);
void findLeftPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d);
void findBottomPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d);
void findTopPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d);

void findTopBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix);
void findBottomBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix);
void findRightBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix);
void findLeftBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix);

void takePrimitivesWithinBoundaries(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *rBnd, int *lBnd, int *bBnd, 
									int *tBnd, MATRIX *primCell, MATRIX tmp);
int computeWidthHeightWithMask(MATRIX sb, int pid, MATRIX mask, int minx, int maxx, int miny, int maxy, int flag);
int computeWidthHeight(MATRIX primCell, int minx, int maxx, int miny, int maxy, int flag);

int computeX3(int x1, int x2, int y1, int y2, double avgR);
int computeY3(int x1, int x2, int y1, int y2, int x3, double avgR);

#endif


