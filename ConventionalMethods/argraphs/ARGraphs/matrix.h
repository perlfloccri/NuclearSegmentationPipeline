#ifndef __matrix_h
#define __matrix_h

#define ALL_COLUMNS -1

struct TMatrixD{
	long row;
	long column;
	double **data;
};
struct TMatrixL{
	long row;
	long column;
	long **data;
};
struct TMatrixC{
	long row;
	long column;
	char **data;
};
struct TMatrix{
	long row;
	long column;
	int **data;
};
typedef struct TMatrixD MATRIXD;
typedef struct TMatrixL MATRIXL;
typedef struct TMatrixC MATRIXC;
typedef struct TMatrix  MATRIX;

MATRIX allocateMatrix(long row, long column);
MATRIXD allocateMatrixD(long row, long column);
MATRIXL allocateMatrixL(long row, long column);
MATRIXC allocateMatrixC(long row, long column);

void reallocateMatrix(MATRIX *M, long row, long column);

void freeMatrix(MATRIX M);
void freeMatrixD(MATRIXD M);
void freeMatrixL(MATRIXL M);
void freeMatrixC(MATRIXC M);

MATRIX readMatrix(char *filename);
MATRIX readMatrixWithoutHeader(char *filename, int colNo);
//MATRIX readMatrixWithoutRowColumnHeader(char *filename);
MATRIXD readMatrixD(char *filename);
MATRIXL readMatrixL(char *filename);

void initializeMatrix(MATRIX *M, int c);
void initializeMatrixD(MATRIXD *M, double c);
void initializeMatrixL(MATRIXL *M, long c);
void initializeMatrixC(MATRIXC *M, char c);
void initializeMatrixPartial(MATRIX *M, int c, int minx, int maxx, int miny, int maxy);

void copyMatrixD(MATRIXD *A, MATRIXD B);
void copyMatrixL(MATRIXL *A, MATRIXL B);
void copyMatrix(MATRIX *A, MATRIX B);
void copyMatrixPartial(MATRIX *A, MATRIX B, int minx, int maxx, int miny, int maxy);
void convertMatrixD(MATRIXD *A, MATRIX B);

MATRIX addMatrix(MATRIX A, MATRIX B);
MATRIXD addMatrixD(MATRIXD A, MATRIXD B);
void incrementMatrix(MATRIX *A, MATRIX B);
void incrementMatrixD(MATRIXD *A, MATRIXD B);
void scalarMatrixAddition(MATRIX *A, int c);
void decrementMatrix(MATRIX *M, int d);

MATRIXD computeMeanMatrixD(MATRIXD A);
MATRIXD computeCovarianceMatrixD(MATRIXD A);
MATRIXD computeCorrelationMatrixD(MATRIXD A);

MATRIXD normalizeMatrixD(MATRIXD *M);
void reverseNormalization(MATRIXD *V, MATRIXD stats);

void displayMatrix(MATRIX M);
void displayMatrixD(MATRIXD M, int precision);
void displayConfusionMatrix(MATRIXD M, int precision);
void writeMatrixIntoFile(MATRIX M, char *filename, int headerFlag);
void writeMatrixDIntoFile(MATRIXD M, char *filename, int precision, int headerFlag);
void writeMatrixLIntoFile(MATRIXL M, char *filename, int headerFlag);
void appendMatrixIntoFile(MATRIX M, char *filename, int headerFlag);
void appendMatrixDIntoFile(MATRIXD M, char *filename, int precision, int headerFlag);

MATRIXD computeEigenValues(MATRIXD M, MATRIXD *V);
MATRIXD inverseMatrixD(MATRIXD M);
MATRIXD multiplyMatrixD(MATRIXD A, MATRIXD B);

int maxMatrixEntry(MATRIX M);
int minMatrixEntry(MATRIX M);
long maxMatrixEntryL(MATRIXL M);
double maxMatrixEntryD(MATRIXD M, long whichColumn);
double minMatrixEntryD(MATRIXD M, long whichColumn);
double maxAbsMatrixDEntry(MATRIXD M);
double maxAbsoluteDifferenceBetweenMatricesD(MATRIXD A, MATRIXD B);
int maxMatrixColumn(MATRIX M, int columnNo);
int minMatrixColumn(MATRIX M, int columnNo);

MATRIX form2DMatrix(MATRIX M, int row, int column);

void replaceParticularMatrixValue(MATRIX *M, int oldValue, int newValue);
long countMatrixOccurrences(MATRIX M, int value);
long countMatrixLOccurrences(MATRIXL M, long value);

long sumOfDifferences(MATRIX A, MATRIX B);

double sumMatrixD(MATRIXD M);

MATRIX matrixTranspose(MATRIX M);

#endif
