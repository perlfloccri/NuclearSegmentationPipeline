#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"
#include "linear.h"
#include "util.h"

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
MATRIXD allocateMatrixD(long row, long column){
	MATRIXD M; 
	long i;
		
	M.row = row; 
	M.column = column;
	M.data = (double **) malloc(row * sizeof(double*));
	for (i = 0; i < row; i++)
		M.data[i] = (double *) malloc(column * sizeof(double));
	return M;
}
MATRIXL allocateMatrixL(long row, long column){
	MATRIXL M; 
	long i;
		
	M.row = row; 
	M.column = column;
	M.data = (long **) malloc(row * sizeof(long*));
	for (i = 0; i < row; i++)
		M.data[i] = (long *) malloc(column * sizeof(long));
	return M;
}
MATRIXC allocateMatrixC(long row, long column){
	MATRIXC M; 
	long i;
		
	M.row = row; 
	M.column = column;
	M.data = (char **) malloc(row * sizeof(char*));
	for (i = 0; i < row; i++)
		M.data[i] = (char *) malloc(column * sizeof(char));
	return M;
}
void reallocateMatrix(MATRIX *M, long row, long column){
	int i, j, minrow, mincol;
	MATRIX tmp = allocateMatrix(M->row,M->column);

	copyMatrix(&tmp,*M);
	freeMatrix(*M);

	*M = allocateMatrix(row,column);
	initializeMatrix(M,0);

	if (tmp.row > row)			minrow = row;
	else						minrow = tmp.row;
	
	if (tmp.column > column)	mincol = column;
	else						mincol = tmp.column;
	
	for (i = 0; i < minrow; i++)
		for (j = 0; j < mincol; j++)
			M->data[i][j] = tmp.data[i][j];
	freeMatrix(tmp);
}
void freeMatrix(MATRIX M){
	long i;
	for (i = 0; i < M.row; i++)
		free(M.data[i]);
	free(M.data);
}
void freeMatrixD(MATRIXD M){
	long i;
	for (i = 0; i < M.row; i++)
		free(M.data[i]);
	free(M.data);
}
void freeMatrixL(MATRIXL M){
	long i;
	for (i = 0; i < M.row; i++)
		free(M.data[i]);
	free(M.data);
}
void freeMatrixC(MATRIXC M){
	long i;
	for (i = 0; i < M.row; i++)
		free(M.data[i]);
	free(M.data);
}
MATRIXD readMatrixD(char *filename){
	MATRIXD M;
	long r, c, i, j;

	FILE *id = fopen(filename,"r");
	if (id == NULL){
		printf("Error: File %s does not exist...\n",filename);
		exit(1);
	}
	fscanf(id,"%ld%ld",&r,&c);
	M = allocateMatrixD(r,c);
	for (i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			fscanf(id,"%lf",&(M.data[i][j]));
	fclose(id);
	return M;
}
MATRIX readMatrix(char *filename){
	MATRIX M;
	long r, c, i, j;

	FILE *id = fopen(filename,"r");
	if (id == NULL){
		printf("Error: File %s does not exist...\n",filename);
		exit(1);
	}
	fscanf(id,"%ld%ld",&r,&c);
	M = allocateMatrix(r,c);
	for (i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			fscanf(id,"%d",&(M.data[i][j]));
	fclose(id);
	return M;
}
MATRIX readMatrixWithoutHeader(char *filename, int colNo){
	MATRIX M;
	FILE *id = fopen(filename,"r");
	int i, j, tmp, rowNo = 0;

	if (id == NULL){
		printf("Error: File %s does not exist...\n",filename);
		exit(1);
	}
	
	while (fscanf(id,"%d",&tmp) != EOF)
		rowNo++;
	rowNo /= colNo;
	fclose(id);

	M = allocateMatrix(rowNo,colNo);
	id = fopen(filename,"r");
	for (i = 0; i < rowNo; i++)
		for (j = 0; j < colNo; j++)
			fscanf(id,"%d",&(M.data[i][j]));
	fclose(id);
	return M;
}
MATRIXL readMatrixL(char *filename){
	MATRIXL M;
	long r, c, i, j;

	FILE *id = fopen(filename,"r");
	if (id == NULL){
		printf("Error: File %s does not exist...\n",filename);
		exit(1);
	}
	fscanf(id,"%ld%ld",&r,&c);
	M = allocateMatrixL(r,c);
	for (i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			fscanf(id,"%ld",&(M.data[i][j]));
	fclose(id);
	return M;
}
void initializeMatrix(MATRIX *M, int c){
	long i, j;
	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] = c;
}
void initializeMatrixD(MATRIXD *M, double c){
	long i, j;
	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] = c;
}
void initializeMatrixL(MATRIXL *M, long c){
	long i, j;
	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] = c;
}
void initializeMatrixC(MATRIXC *M, char c){
	long i, j;
	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] = c;
}
void initializeMatrixPartial(MATRIX *M, int c, int minx, int maxx, int miny, int maxy){
	long i, j;
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			M->data[i][j] = c;
}
MATRIXD normalizeMatrixD(MATRIXD *M){
	MATRIXD stats = allocateMatrixD(2,M->column);
	long i, j;
	
	for (i = 0; i < M->column; i++){
		stats.data[0][i] = 0.0;
		for (j = 0; j < M->row; j++)
			stats.data[0][i] += M->data[j][i];
		stats.data[0][i] /= M->row;

		stats.data[1][i] = 0;
		for (j = 0; j < M->row; j++)
			stats.data[1][i] += (M->data[j][i] - stats.data[0][i]) * (M->data[j][i] - stats.data[0][i]);
		stats.data[1][i] /= M->row - 1;
		stats.data[1][i] = sqrt(stats.data[1][i]);
	}

	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			if (stats.data[1][j] <= ZERO)
				M->data[i][j] = (M->data[i][j] - stats.data[0][j]); 
			else
				M->data[i][j] = (M->data[i][j] - stats.data[0][j]) / stats.data[1][j]; 
	return stats;
}
void reverseNormalization(MATRIXD *V, MATRIXD stats){
	long i, j;
	for (i = 0; i < V->row; i++)
		for (j = 0; j < V->column; j++)
			V->data[i][j] = V->data[i][j] * stats.data[1][j] + stats.data[0][j];
}
void displayConfusionMatrix(MATRIXD M, int precision){
    long i, j;
	char temp[100];

	sprintf(temp,"%c.%d%c\n",37,precision,'f');
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column - 1; j++)
			printf("%.0lf\t",M.data[i][j]);
		printf(temp,M.data[i][j]);
	}
}
void displayMatrixD(MATRIXD M, int precision){
    long i, j;
	char temp[100];

	sprintf(temp,"%c.%d%c ",37,precision,'f');
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			printf(temp,M.data[i][j]);
		printf("\n");
	}
}
void displayMatrix(MATRIX M){
    long i, j;

	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			printf("%d ",M.data[i][j]);
		printf("\n");
	}
}
void writeMatrixDIntoFile(MATRIXD M, char *filename, int precision, int headerFlag){
    long i, j;
	char temp[100];
    FILE *id = fopen(filename,"w");
    
	if (headerFlag)
		fprintf(id,"%ld\t%ld\n",M.row,M.column);
	sprintf(temp,"%c.%d%c ",37,precision,'f');
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			fprintf(id,temp,M.data[i][j]);
		fprintf(id,"\n");
	}
	fclose(id);
}
void appendMatrixDIntoFile(MATRIXD M, char *filename, int precision, int headerFlag){
    long i, j;
	char temp[100];
    FILE *id = fopen(filename,"a");
    
	if (headerFlag)
		fprintf(id,"%ld\t%ld\n",M.row,M.column);
	sprintf(temp,"%c.%d%c ",37,precision,'f');
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			fprintf(id,temp,M.data[i][j]);
		fprintf(id,"\n");
	}
	fclose(id);
}
void writeMatrixIntoFile(MATRIX M, char *filename, int headerFlag){
    long i, j;
    FILE *id = fopen(filename,"w");
    
	if (headerFlag)
		fprintf(id,"%ld\t%ld\n",M.row,M.column);
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			fprintf(id,"%d ",M.data[i][j]);
		fprintf(id,"\n");
	}
	fclose(id);
}
void appendMatrixIntoFile(MATRIX M, char *filename, int headerFlag){
    long i, j;
    FILE *id = fopen(filename,"a");
    
	if (headerFlag)
		fprintf(id,"%ld\t%ld\n",M.row,M.column);
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			fprintf(id,"%d ",M.data[i][j]);
		fprintf(id,"\n");
	}
	fclose(id);
}
void writeMatrixLIntoFile(MATRIXL M, char *filename, int headerFlag){
    long i, j;
    FILE *id = fopen(filename,"w");
    
	if (headerFlag)
		fprintf(id,"%ld\t%ld\n",M.row,M.column);
	for (i = 0; i < M.row; i++){
		for (j = 0; j < M.column; j++)
			fprintf(id,"%ld ",M.data[i][j]);
		fprintf(id,"\n");
	}
	fclose(id);
}
double maxAbsMatrixDEntry(MATRIXD M){
	double maxAbs;
	long i, j;

	maxAbs = fabs(M.data[0][0]);
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (fabs(M.data[i][j]) > maxAbs)
				maxAbs = fabs(M.data[i][j]);
	return maxAbs;
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
int minMatrixEntry(MATRIX M){
	int minEntry;
	long i, j;

	minEntry = M.data[0][0];
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (minEntry > M.data[i][j])
				minEntry = M.data[i][j];
	return minEntry;
}
long maxMatrixEntryL(MATRIXL M){
	long maxEntry, i, j;

	maxEntry = M.data[0][0];
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (maxEntry < M.data[i][j])
				maxEntry = M.data[i][j];
	return maxEntry;
}
int maxMatrixColumn(MATRIX M, int columnNo){
	int maxCol, i;

	if (columnNo >= M.column)
		terminateProgram("Error in maxMatrixColumn (matrix.c)");
	
	maxCol = M.data[0][columnNo];
	for (i = 0; i < M.row; i++)
		if (M.data[i][columnNo] > maxCol)
			maxCol = M.data[i][columnNo];

	return maxCol;
}
int minMatrixColumn(MATRIX M, int columnNo){
	int minCol, i;

	if (columnNo >= M.column)
		terminateProgram("Error in maxMatrixColumn (matrix.c)");
	
	minCol = M.data[0][columnNo];
	for (i = 0; i < M.row; i++)
		if (M.data[i][columnNo] < minCol)
			minCol = M.data[i][columnNo];

	return minCol;
}
void copyMatrixD(MATRIXD *A, MATRIXD B){
	long i, j;
	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in copy operation\n");
		exit(1);
	}
	for (i = 0; i < B.row; i++)
		for (j = 0; j < B.column; j++)
			A->data[i][j] = B.data[i][j];
}
void copyMatrix(MATRIX *A, MATRIX B){
	long i, j;
	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in copy operation\n");
		exit(1);
	}
	for (i = 0; i < B.row; i++)
		for (j = 0; j < B.column; j++)
			A->data[i][j] = B.data[i][j];
}
void copyMatrixPartial(MATRIX *A, MATRIX B, int minx, int maxx, int miny, int maxy){
	long i, j;
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			A->data[i][j] = B.data[i][j];
}
void copyMatrixL(MATRIXL *A, MATRIXL B){
	long i, j;
	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in copy operation\n");
		exit(1);
	}
	for (i = 0; i < B.row; i++)
		for (j = 0; j < B.column; j++)
			A->data[i][j] = B.data[i][j];
}
void convertMatrixD(MATRIXD *A, MATRIX B){
	long i, j;

	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in copy operation\n");
		exit(1);
	}
	for (i = 0; i < B.row; i++)
		for (j = 0; j < B.column; j++)
			A->data[i][j] = B.data[i][j];
}
MATRIX addMatrix(MATRIX A, MATRIX B){
	long i, j;
	MATRIX M;

	if (A.row != B.row || A.column != B.column){
		printf("\nError: Matrix dimensions mismatch in add operation\n");
		exit(1);
	}
	M = allocateMatrix(A.row,A.column);
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			M.data[i][j] = A.data[i][j] + B.data[i][j];
	return M;
}
MATRIXD addMatrixD(MATRIXD A, MATRIXD B){
	long i, j;
	MATRIXD M;

	if (A.row != B.row || A.column != B.column){
		printf("\nError: Matrix dimensions mismatch in add operation\n");
		exit(1);
	}
	M = allocateMatrixD(A.row,A.column);
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			M.data[i][j] = A.data[i][j] + B.data[i][j];
	return M;
}
void incrementMatrix(MATRIX *A, MATRIX B){
	long i, j;

	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in add operation\n");
		exit(1);
	}
	for (i = 0; i < A->row; i++)
		for (j = 0; j < A->column; j++)
			A->data[i][j] += B.data[i][j];
}
void incrementMatrixD(MATRIXD *A, MATRIXD B){
	long i, j;

	if (A->row != B.row || A->column != B.column){
		printf("\nError: Matrix dimensions mismatch in add operation\n");
		exit(1);
	}
	for (i = 0; i < A->row; i++)
		for (j = 0; j < A->column; j++)
			A->data[i][j] += B.data[i][j];
}
void scalarMatrixAddition(MATRIX *A, int c){
	long i, j;

	for (i = 0; i < A->row; i++)
		for (j = 0; j < A->column; j++)
			A->data[i][j] += c;
}
MATRIXD computeMeanMatrixD(MATRIXD A){
	long i, j;
	MATRIXD M = allocateMatrixD(1,A.column);

	initializeMatrixD(&M,0.0);
	for (i = 0; i < A.row; i++)
		for (j = 0; j < A.column; j++)
			M.data[0][j] += A.data[i][j];
	for (j = 0; j < M.column; j++)
			M.data[0][j] /= A.row;
	return M;
}
MATRIXD computeCovarianceMatrixD(MATRIXD A){
	long i, j, t;
	MATRIXD M = computeMeanMatrixD(A);
	MATRIXD S = allocateMatrixD(A.column,A.column);

	initializeMatrixD(&S,0.0);
	for (i = 0; i < S.row; i++)
		for (j = 0; j < S.column; j++)
			for (t = 0; t < A.row; t++)
				S.data[i][j] += (A.data[t][i] - M.data[0][i]) * (A.data[t][j] - M.data[0][j]);
	for (i = 0; i < S.row; i++)
		for (j = 0; j < S.column; j++)
			S.data[i][j] /= A.row - 1;	
	freeMatrixD(M);
	return S;
}
MATRIXD computeCorrelationMatrixD(MATRIXD A){
	long i, j;
	MATRIXD S = computeCovarianceMatrixD(A);
	MATRIXD R = allocateMatrixD(S.row,S.column);
	for (i = 0; i < S.row; i++)
		for (j = 0; j < S.column; j++)
			if (fabs(sqrt(S.data[i][i]) * sqrt(S.data[j][j])) > ZERO)
				R.data[i][j] = S.data[i][j] / (sqrt(S.data[i][i]) * sqrt(S.data[j][j]));
			else
				R.data[i][j] = 0;
	freeMatrixD(S);
	return R;
}
// for symmetric matrices
MATRIXD computeEigenValues(MATRIXD M, MATRIXD *V){
	long i, j;
	int nrot;
	MATRIXD D;
	
	if (M.row != M.column){
		printf("\nError: Matrix should be square for eigenvalue computation\n");
		exit(1);
	}
	for (i = 0; i < M.row; i++)
		for (j = i+1; j < M.column; j++)
			if (M.data[i][j] != M.data[j][i]){
				printf("\nError: Matrix should be symmetric for eigenvalue computation\n");
				exit(1);
			}
	D = allocateMatrixD(1,M.row);
	*V = allocateMatrixD(M.row,M.row);
	jacobi(M.data,M.row,D.data[0],V->data,&nrot);
	eigsrt(D.data[0],V->data,D.column);
	return D;
}
MATRIXD inverseMatrixD(MATRIXD M){
	MATRIXD inv, temp;
	int i;
	
	if (M.row != M.column){
		printf("\nError: Matrix should be square for inverse computation\n");
		exit(1);
	}
	temp = allocateMatrixD(M.row,M.column);
	copyMatrixD(&temp,M);
	inv = allocateMatrixD(M.row,M.column);
	initializeMatrixD(&inv,0.0);
	for (i = 0; i < M.row; i++)
		inv.data[i][i] = 1;
	gaussj(M.data,M.row,inv.data,inv.column);
	copyMatrixD(&M,temp);
	freeMatrixD(temp);
	return inv;
}
MATRIXD multiplyMatrixD(MATRIXD A, MATRIXD B){
	MATRIXD result;
	long i, j, k;

	if (A.column != B.row){
		printf("\nError: Matrix dimensions do not match in matrix multiplication\n");
		exit(1);
	}
	result = allocateMatrixD(A.row,B.column);
	for (i = 0; i < A.row; i++)
		for (j = 0; j < B.column; j++){
			result.data[i][j] = 0.0;
			for (k = 0; k < A.column; k++)
				result.data[i][j] += A.data[i][k] * B.data[k][j];
		}
	return result;
}
double maxMatrixEntryD(MATRIXD M, long whichColumn){
	double maxEntry;
	long i, j;

	if (whichColumn == ALL_COLUMNS){
		maxEntry = M.data[0][0];
		for (i = 0; i < M.row; i++)
			for (j = 0; j < M.column; j++)
				if (maxEntry < M.data[i][j])
					maxEntry = M.data[i][j];
		return maxEntry;
	}
	maxEntry = M.data[0][whichColumn];
	for (i = 0; i < M.row; i++)
		if (maxEntry < M.data[i][whichColumn])
			maxEntry = M.data[i][whichColumn];
	return maxEntry;
}
double minMatrixEntryD(MATRIXD M, long whichColumn){
	double minEntry;
	long i, j;

	if (whichColumn == ALL_COLUMNS){
		minEntry = M.data[0][0];
		for (i = 0; i < M.row; i++)
			for (j = 0; j < M.column; j++)
				if (minEntry > M.data[i][j])
					minEntry = M.data[i][j];
		return minEntry;
	}
	minEntry = M.data[0][whichColumn];
	for (i = 0; i < M.row; i++)
		if (minEntry > M.data[i][whichColumn])
			minEntry = M.data[i][whichColumn];
	return minEntry;
}
double maxAbsoluteDifferenceBetweenMatricesD(MATRIXD A, MATRIXD B){
	long i, j;
	double maxAbsEntry = fabs(A.data[0][0] - B.data[0][0]);

	for (i = 0; i < A.row; i++)
		for (j = 0; j < A.column; j++)
			if (maxAbsEntry < fabs(A.data[i][j] - B.data[i][j]))
				maxAbsEntry = fabs(A.data[i][j] - B.data[i][j]);
	return maxAbsEntry;
}
MATRIX form2DMatrix(MATRIX M, int row, int column){
	MATRIX newM;
	int i, j, c;

	if (row * column != M.row){
		printf("\nError: dimensions mismatch\n\n");
		exit(1);
	}
	newM = allocateMatrix(row,column);
	c = 0;
	for (i = 0; i < row; i++)
		for (j = 0; j < column; j++)
			newM.data[i][j] = M.data[c++][0];
	return newM;
}
void replaceParticularMatrixValue(MATRIX *M, int oldValue, int newValue){
	long x, y;

	for (x = 0; x < M->row; x++)
		for (y = 0; y < M->column; y++)
			if (M->data[x][y] == oldValue)
				M->data[x][y] = newValue;
}
long countMatrixOccurrences(MATRIX M, int value){
	long occ = 0, i, j;

	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (M.data[i][j] == value)
				occ++;
	return occ;
}
long countMatrixLOccurrences(MATRIXL M, long value){
	long occ = 0, i, j;

	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			if (M.data[i][j] == value)
				occ++;
	return occ;
}
long sumOfDifferences(MATRIX A, MATRIX B){
	long i, j;
	long d = 0;

	if (A.row != B.row || A.column != B.column){
		printf("\nError: Matrix dimensions mismatch in sum of differences operation\n");
		exit(1);
	}
	for (i = 0; i < A.row; i++)
		for (j = 0; j < A.column; j++)
			if (A.data[i][j] != B.data[i][j])
				d++;
	return d;
}
double sumMatrixD(MATRIXD M){
	double sum = 0.0;
	int i, j;

	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			sum += M.data[i][j];
	return sum;
}
void decrementMatrix(MATRIX *M, int d){
	int i, j;

	for (i = 0; i < M->row; i++)
		for (j = 0; j < M->column; j++)
			M->data[i][j] -= d;
}
MATRIX matrixTranspose(MATRIX M){
	MATRIX T = allocateMatrix(M.column,M.row);
	int i, j;
	
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++)
			T.data[j][i] = M.data[i][j];
	
	return T;
}




