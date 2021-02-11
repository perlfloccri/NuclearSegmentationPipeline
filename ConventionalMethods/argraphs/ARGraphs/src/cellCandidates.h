#ifndef __cellCandidates_h
#define __cellCandidates_h

#define START_TO_NO	1
#define NO_TO_END	2

struct TCellCandidate{
	int ids[4];
	int ctype;
	double cx, cy;
	double avgR, stdR;
	int perimeter;
	int taken;
};
typedef struct TCellCandidate CANDIDATE;
 
struct TCandidates{
	int maxNo; 
	int cellNo;
	int dx, dy;
	CANDIDATE *cells;
};
typedef struct TCandidates CELLS;

///******************************************************************/
///******************************************************************/
///******************************************************************/
CELLS allocateCellCandidates(int maxNo, int dx, int dy);
void freeCellCandidates(CELLS C);
void extendCellCandidates(CELLS *C, int newSize);
void add2CellCandidates(CELLS *C, MATRIX aCell, int ids[4], int minx, int maxx, int miny, int maxy, int ctype);

void fillInsideCell(MATRIX *aCell, int label, int minx, int maxx, int miny, int maxy, MATRIX res);
///******************************************************************/
///******************************************************************/
///******************************************************************/
void markRightOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy);
void markLeftOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy);
void markTopOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy);
void markBottomOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy);

int getRightOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy);
int getLeftOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy);
int getTopOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy);
int getBottomOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy);
///******************************************************************/
///******************************************************************/
///******************************************************************/
int arePrimsNeighbor(MATRIX aCell, MATRIX p1, MATRIX p2, int id1, int id2, int minx, int maxx, int miny, int maxy);
int arePrimCellsNeighbor(MATRIX p1, MATRIX p2, int minx, int maxx, int miny, int maxy);
void computeIncrementalCentroid(int *bndx, int *bndy, int no, double *cx, double *cy, int *cno);
double computeCentroidRadius(MATRIX aCell, int minx, int maxx, int miny, int maxy, double *cx, double *cy, int *cno);
int getRangePoints(int x, int y, double cx, double cy, int *bndx, int *bndy, int no);
void getClosestBoundaryPoints(int *bndx1, int *bndy1, int n1, int typ1, int *bndx2, int *bndy2, int n2, int typ2, 
							  double cx, double cy, int *p1, int *p2);
void getClosestBoundaryPointsLeftRight(int *bndx1, int *bndy1, int n1, int *bndx2, int *bndy2, int n2,
									   double cx, double cy, int *p1, int *p2, int flag);
void getClosestBoundaryPointsTopBottom(int *bndx1, int *bndy1, int n1, int *bndx2, int *bndy2, int n2,
									   double cx, double cy, int *p1, int *p2, int flag);
void clearNoisyOutsidePrimitives(MATRIX *aCell, MATRIX *primCell, int minx, int maxx, int miny, int maxy);
void swapPoints(int *x1, int *y1, int *x2, int *y2);
void completeCell(MATRIX *aCell, int x1, int y1, int x2, int y2, double cx, double cy, double avgR, int inc, int *minx, int *maxx, int *miny, int *maxy);
void setCell(MATRIX *aCell, int minx, int maxx, int miny, int maxy);
///******************************************************************/
///******************************************************************/
///******************************************************************/
void markOuterPixels(MATRIX *M, int minx, int maxx, int miny, int maxy);
int computeCellOuterCentroid(MATRIX aCell, int minx, int maxx, int miny, int maxy, double *cx, double *cy);
void computeAvgStdRadius(MATRIX aCell, int minx, int maxx, int miny, int maxy, int cnt, double cx, double cy, double *avgR, double *stdR);

///******************************************************************/
///******************************************************************/
///******************************************************************/
void eliminateTopPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp);
void eliminateBottomPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp);
void eliminateRightPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp);
void eliminateLeftPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp);
void eliminatePrimitivesBasedOnResults(MATRIX bnd, MATRIX *sb, int stype, MATRIX res, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp);

void erodeCell(MATRIX *aCell, MATRIX se, int minx, int maxx, int miny, int maxy, MATRIX res);

#endif


