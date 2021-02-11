#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"

int indexOfMinArrayEntry(double A[], int size){
	int minIndex = 0, i;
	for (i = 1; i < size; i++)
		if (A[i] < A[minIndex])
			minIndex = i;
	return minIndex;
}
int maxArrayEntryIndex(int *A, int size){
	int i, maxIndex = 0;

	for (i = 0; i < size; i++)
		if (A[i] > A[maxIndex])
			maxIndex = i;
	return maxIndex;
}
int *readIntArrayFromFile(char *filename, int *no){
	FILE *id = fopen(filename,"r");
	int *A, i;

	fscanf(id,"%d",no);
	A = (int *)calloc(*no,sizeof(int));
	for (i = 0; i < *no; i++)
		fscanf(id,"%d",&(A[i]));

	fclose(id);
	return A;
}
void writeDoubleArrayIntoFile(double *A, long size, char *filename){
	long i;
	FILE *id = fopen(filename,"w");
	
	for (i = 0; i < size; i++)
		fprintf(id,"%lf\n",A[i]);
	fclose(id);
}
void writeIntArrayIntoFile(int *A, long size, char *filename){
	long i;
	FILE *id = fopen(filename,"w");
	
	for (i = 0; i < size; i++)
		fprintf(id,"%d\n",A[i]);
	fclose(id);
}
void sortDouble(double *A, int size, int *aOrder){
	long i, j, temp;

	for (i = 0; i < size; i++)
		aOrder[i] = i;
	for (i = 1; i < size; i++){
		temp = aOrder[i];
		for (j = i - 1; j >= 0; j--){
			if (A[ aOrder[j] ] <= A[temp])
				break;
			aOrder[j+1] = aOrder[j];
		}
		aOrder[j+1] = temp;
	}
}
void sortLong(long *A, int size, int *aOrder){
	long i, j, temp;

	for (i = 0; i < size; i++)
		aOrder[i] = i;
	for (i = 1; i < size; i++){
		temp = aOrder[i];
		for (j = i - 1; j >= 0; j--){
			if (A[ aOrder[j] ] <= A[temp])
				break;
			aOrder[j+1] = aOrder[j];
		}
		aOrder[j+1] = temp;
	}
}
double averageArray(double *A, int no){
	double avg = 0.0;
	int i;

	for (i = 0; i < no; i++)
		avg += A[i];
	return (avg / no);
}
double varianceArray(double *A, int no){
	double avg, var;
	int i;

	avg = var = 0.0;
	for (i = 0; i < no; i++)
		avg += A[i];
	avg /= no;

	for (i = 0; i < no; i++)
		var += (A[i] - avg) * (A[i] - avg);
	if (no - 1 > 0)
		var /= (no - 1);

	return var;
}
void terminateProgram(char *str){
	printf("\n%s\n\n",str);
	exit(1);
}
int computeAngle(double x1, double y1, double x2, double y2, double *d){
	double tmp;

	if ((fabs(x1 - x2) <= ZERO) && (fabs(y1 - y2) <= ZERO)){
		//printf("\nSame points\n\n\n");
		*d = 0.0;
		return 0;
	}
	if (fabs(y1 - y2) <= ZERO){
		if (x2 > x1)		tmp = PI;
		else				tmp = 0.0;
	}
	else if (fabs(x1 - x2) <= ZERO){
		if (y2 > y1)		tmp = PI + PI/2;
		else				tmp = PI/2;
	}
	else{
		tmp = fabs(atan((y1-y2)/(x1-x2)));
		if ((y1-y2 > 0) && (x1-x2 > 0))				tmp = tmp;
		else if ((y1-y2 < 0) && (x1-x2 < 0))		tmp = PI + tmp;
		else if ((y1-y2 > 0) && (x1-x2 < 0))		tmp = PI - tmp;
		else if ((y1-y2 < 0) && (x1-x2 > 0))		tmp = 2*PI - tmp;
	}
	*d = (tmp * 180) / PI;
	return 1;
}
int findMedian(int A[], int no){
	int midP = no / 2, i, k, largestInd, median = -1;
	
	for (k = 0; k < midP; k++){
		largestInd = 0;
		for (i = 1; i < no; i++)
			if (A[largestInd] < A[i])
				largestInd = i;
		if (k == midP - 1)
			median = A[largestInd];
		A[largestInd] = A[no - 1];
		no--;
	}
	return median;
}
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
	child = 2 * root + 1; // left child, if any
	while (child < H->size){
		rchild = child + 1; // right child
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
