#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"

/*********************************************************/
/****************** general definitions ******************/
/*********************************************************/
void terminateProgram(char *str){
	printf("\n%s\n\n",str);
	exit(1);
}

/*********************************************************/
/****************** heap data structure ******************/
/*********************************************************/
HEAP initializeHeap(int maxSize, int typ){
	HEAP H;
	H.size = 0;
	H.maxSize = maxSize;
	H.data = (HEAPDATA *)calloc(maxSize,sizeof(HEAPDATA));
	H.typ = typ;
	return H;
}
void freeHeap(HEAP H){
	free(H.data);
}
void insertHeap(HEAP *H, double key, int cx, int cy, int label){
	int place, parent;
	HEAPDATA tmp;
	
	if (H->size >= H->maxSize)
		terminateProgram("Cannot insert to a full heap");
	
	H->data[H->size].key = key;
	H->data[H->size].cx = cx;
	H->data[H->size].cy = cy;
	H->data[H->size].label = label;
    
	place = H->size;
	parent = (place - 1) / 2;
	
	while (place > 0){
		if ((H->typ == MAXHEAP) && (H->data[place].key < H->data[parent].key))
			break;
		if ((H->typ == MINHEAP) && (H->data[place].key > H->data[parent].key))
			break;
        
		tmp = H->data[place];
		H->data[place] = H->data[parent];
		H->data[parent] = tmp;
		
		place = parent;
		parent = (place - 1) / 2;
	}
	H->size += 1;
}
HEAPDATA deleteHeap(HEAP *H){
	HEAPDATA item, tmp;
	int root, child, rchild;
	
	if (H->size == 0)
		terminateProgram("Cannot delete from an empty heap");
	
	item = H->data[0];
	H->size -= 1;
	H->data[0] = H->data[H->size];
	
	root = 0;
	child = 2 * root + 1;
	while (child < H->size){
		rchild = child + 1;
		if (rchild < H->size){
			if ((H->typ == MAXHEAP) && (H->data[rchild].key > H->data[child].key))
				child = rchild;
			if ((H->typ == MINHEAP) && (H->data[rchild].key < H->data[child].key))
				child = rchild;
		}
		if ((H->typ == MAXHEAP) && (H->data[root].key > H->data[child].key))
			break;
		if ((H->typ == MINHEAP) && (H->data[root].key < H->data[child].key))
			break;
		
		tmp = H->data[root];
		H->data[root] = H->data[child];
		H->data[child] = tmp;
		root = child;
		child = 2 * root + 1;
	}
	return item;
}

/*********************************************************/
/****** bounding box definition for fast computation *****/
/*********************************************************/
BNDBOX *calculateBoundingBoxes(MATRIX reg, int background){
	int regNo = maxMatrixEntry(reg), i, j, rno;
	BNDBOX *bnd = (BNDBOX *)calloc(regNo+1,sizeof(BNDBOX));
    
	for (i = 0; i <= regNo; i++)
		bnd[i].minx = bnd[i].maxx = bnd[i].miny = bnd[i].maxy = -1;
    
	for (i = 0; i < reg.row; i++)
		for (j = 0; j < reg.column; j++)
			if (reg.data[i][j] != background){
				rno = reg.data[i][j];
				if ((bnd[rno].minx == -1) || (i < bnd[rno].minx))		bnd[rno].minx = i;
				if ((bnd[rno].maxx == -1) || (i > bnd[rno].maxx))		bnd[rno].maxx = i;
				if ((bnd[rno].miny == -1) || (j < bnd[rno].miny))		bnd[rno].miny = j;
				if ((bnd[rno].maxy == -1) || (j > bnd[rno].maxy))		bnd[rno].maxy = j;
			}
	return bnd;
}

/***************************************************************/
/*********** used for efficient distance calculation ***********/
/***************************************************************/
CIRCLE_BND *createCircularBoundaries(int maxRadius){
	CIRCLE_BND *B = (CIRCLE_BND *)calloc(maxRadius,sizeof(CIRCLE_BND));
	MATRIX M = allocateMatrix(maxRadius,maxRadius);
	int i, j, k, cnt;
    
	initializeMatrix(&M,-1);
	
	M.data[0][0] = 0;
	B[0].N = 1;
	B[0].r = 0;
	B[0].x = (int *)calloc(1,sizeof(int));
	B[0].y = (int *)calloc(1,sizeof(int));
	B[0].x[0] = 0;
	B[0].y[0] = 0;
    
	for (k = 1; k < maxRadius; k++){
		B[k].r = k;
		B[k].N = 0;
		B[k].x = B[k].y = NULL;
		for (i = 0; i <= k; i++)
			for (j = 0; j <= k; j++){
				if (M.data[i][j] != -1)
					continue;
				if (i*i + j*j <= k*k){
					M.data[i][j] = k;
					B[k].N += 1;
				}
			}
		if (B[k].N == 0)
			continue;
		B[k].x = (int *)calloc(B[k].N,sizeof(int));
		B[k].y = (int *)calloc(B[k].N,sizeof(int));
		cnt = 0;
		for (i = 0; i <= k; i++)
			for (j = 0; j <= k; j++)
				if (M.data[i][j] == k){
					B[k].x[cnt] = i;
					B[k].y[cnt] = j;
					cnt++;
				}
	}
	freeMatrix(M);
	return B;
}
void freeCircularBoundaries(CIRCLE_BND *B, int maxRadius){
	int i;
    
	for (i = 0; i < maxRadius; i++){
		free(B[i].x);
		free(B[i].y);
	}
	free(B);
}
