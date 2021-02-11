#ifndef __MATRIX_H
#define __MATRIX_H

#include <mex.h>

/*********************************************************/
/***************** matrix data structure *****************/
/*********************************************************/
struct TMatrix{
	long row;
	long column;
	int **data;
};
typedef struct TMatrix  MATRIX;

MATRIX allocateMatrix(long row, long column);
void freeMatrix(MATRIX M);
void initializeMatrix(MATRIX *M, int c);
MATRIX  convertMxArray2Matrix(const mxArray *xData);
mxArray *convertMatrix2MxArray(MATRIX M);
int maxMatrixEntry(MATRIX M);
#endif
