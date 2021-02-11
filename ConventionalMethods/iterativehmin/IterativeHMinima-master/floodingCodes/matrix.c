#include <math.h>
#include <mex.h>
#include "matrix.h"


/*********************************************************/
/***************** matrix data structure *****************/
/*********************************************************/
MATRIX allocateMatrix(long row, long column){
	MATRIX M;
	long i;
    
	M.row = row;
	M.column = column;
	M.data = (int **) malloc(row * sizeof(int*));
	for (i = 0; i < row; i++)
		M.data[i] = (int *) malloc(column * sizeof(int));
	return M;
}
void freeMatrix(MATRIX M){
	long i;
	for (i = 0; i < M.row; i++)
		free(M.data[i]);
	free(M.data);
}
void initializeMatrix(MATRIX *M, int c){
	long i, j;
	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] = c;
}

MATRIX convertMxArray2Matrix(const mxArray *xData){
	double *xValues = mxGetPr(xData);
	int row = mxGetM(xData);
	int col = mxGetN(xData);
	MATRIX res = allocateMatrix(row,col);
	int i, j, cnt = 0;
	
	for (j = 0; j < col; j++)
		for (i = 0; i < row; i++)
			res.data[i][j] = (int)xValues[cnt++];	
	
	return res;
}

mxArray *convertMatrix2MxArray(MATRIX M){
	mxArray *res = mxCreateDoubleMatrix(M.row,M.column,mxREAL);
	double *xValues = mxGetPr(res);	
	int i, j, cnt = 0;
	
	for (j = 0; j < M.column; j++)
		for (i = 0; i < M.row; i++)
			xValues[cnt++] = M.data[i][j];

	return res;
}

int maxMatrixEntry(MATRIX M){
	int maxEntry;
	long i, j;

	maxEntry = M.data[0][0];
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (maxEntry < M.data[i][j])
				maxEntry = M.data[i][j];
	return maxEntry;
}
