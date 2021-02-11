#ifndef __UTIL_H
#define __UTIL_H

#include "matrix.h"

/*********************************************************/
/****************** general definitions ******************/
/*********************************************************/
#define SQUARE(a) ( (a) * (a) )
void terminateProgram(char *str);

/*********************************************************/
/****************** heap data structure ******************/
/*********************************************************/
#define MAXHEAP 1
#define MINHEAP 2
#define PI	3.14159265358979

struct THeapData {
	double key;
	int cx, cy;
	int label;
};
typedef struct THeapData HEAPDATA;

struct THeap{
	HEAPDATA *data;
	int maxSize;
	int size;
	int typ;
};
typedef struct THeap HEAP;

HEAP initializeHeap(int maxSize, int typ);
void freeHeap(HEAP H);
void insertHeap(HEAP *H, double key, int cx, int cy, int label);
HEAPDATA deleteHeap(HEAP *H);

/*********************************************************/
/****** bounding box definition for fast computation *****/
/*********************************************************/
struct TBoundingBox{
	int minx, maxx, miny, maxy;
};
typedef struct TBoundingBox BNDBOX;
BNDBOX *calculateBoundingBoxes(MATRIX reg, int background);

/***************************************************************/
/*********** used for efficient distance calculation ***********/
/***************************************************************/
struct TCircleBoundaries{
	int *x, *y;
	int N, r;
};
typedef struct TCircleBoundaries CIRCLE_BND;
CIRCLE_BND *createCircularBoundaries(int maxRadius);
void freeCircularBoundaries(CIRCLE_BND *B, int maxRadius);

#endif


