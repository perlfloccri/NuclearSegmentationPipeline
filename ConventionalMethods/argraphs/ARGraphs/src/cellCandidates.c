#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../util.h"
#include "../matrix.h"
#include "../imageProcessing.h"
#include "cellCandidates.h"
#include "primitiveOperations.h"


CELLS allocateCellCandidates(int maxNo, int dx, int dy){
	CELLS C;
	int i;
	
	C.dx = dx;
	C.dy = dy;
	C.maxNo = maxNo;
	C.cellNo = 0;
	C.cells = (CANDIDATE *)calloc(maxNo,sizeof(CANDIDATE));
	
	return C;
}
void freeCellCandidates(CELLS C){

	free(C.cells);
}
void extendCellCandidates(CELLS *C, int newSize){
	CANDIDATE *tmp = C->cells;
	int i, j;
	
	if (newSize < C->maxNo)
		terminateProgram("You cannot shrink the size of the candidates array");
	
	C->maxNo = newSize;
	C->cells = (CANDIDATE *)calloc(newSize,sizeof(CANDIDATE));
	
	for (i = 0; i < C->cellNo; i++){
		C->cells[i] = tmp[i];
		for (j = 0; j < 4; j++)
			C->cells[i].ids[j] = tmp[i].ids[j];
	}
	free(tmp);
}
void markOuterPixels(MATRIX *M, int minx, int maxx, int miny, int maxy){
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++){
			if (M->data[i][j] == 0)
				continue;
			
			if ((i == minx) || (M->data[i-1][j] == 0))				M->data[i][j] = 2;
			else if ((i == maxx) || (M->data[i+1][j] == 0))			M->data[i][j] = 2;
			else if ((j == miny) || (M->data[i][j-1] == 0))			M->data[i][j] = 2;
			else if ((j == maxy) || (M->data[i][j+1] == 0))			M->data[i][j] = 2;
		}
}
int computeCellOuterCentroid(MATRIX aCell, int minx, int maxx, int miny, int maxy, double *cx, double *cy){
	int i, j, cnt = 0;
	
	*cx = *cy = 0.0;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (aCell.data[i][j] == 2){
				cnt++;
				(*cx) += i;
				(*cy) += j;
			}
	
	if (cnt){
		(*cx) /= cnt;
		(*cy) /= cnt;
	}
	return cnt;
}
void computeAvgStdRadius(MATRIX aCell, int minx, int maxx, int miny, int maxy, int cnt, double cx, double cy, double *avgR, double *stdR){
	double *radius;
	int i, j;
	
	*avgR = *stdR = 0.0;
	if (cnt == 0)
		return;
	
	radius = (double *)calloc(cnt,sizeof(double));
	cnt = 0;
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (aCell.data[i][j] == 2){
				radius[cnt] = sqrt((cx - i)*(cx - i) + (cy - j)*(cy - j));
				(*avgR) += radius[cnt];
				cnt++;
			}
	(*avgR) /= cnt;
	
	for (i = 0; i < cnt; i++)
		(*stdR) += (radius[i] - (*avgR)) * (radius[i] - (*avgR));
	(*stdR) = sqrt((*stdR) / (cnt - 1));
	
	free(radius);
}
void add2CellCandidates(CELLS *C, MATRIX aCell, int ids[4], int minx, int maxx, int miny, int maxy, int ctype){
	int cid = C->cellNo;
	
	if (C->cellNo + 1 == C->maxNo)
		extendCellCandidates(C,2 * C->maxNo);
	
	C->cells[cid].ctype = ctype;
	
	C->cells[cid].ids[RIGHT] = ids[RIGHT];
	C->cells[cid].ids[LEFT] = ids[LEFT];
	C->cells[cid].ids[TOP] = ids[TOP];
	C->cells[cid].ids[BOTTOM] = ids[BOTTOM];
	
	markOuterPixels(&aCell,minx,maxx,miny,maxy);
	C->cells[cid].perimeter = computeCellOuterCentroid(aCell,minx,maxx,miny,maxy,&(C->cells[cid].cx),&(C->cells[cid].cy));
	computeAvgStdRadius(aCell,minx,maxx,miny,maxy,C->cells[cid].perimeter,C->cells[cid].cx,C->cells[cid].cy,&(C->cells[cid].avgR),&(C->cells[cid].stdR));
	
	C->cells[cid].taken = 0;
	C->cellNo += 1;
}
int getRightOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy){

	int x, y, prevY = -1, i, no = 0;
	
	for (x = minx; x <= maxx; x++)
		for (y = maxy; y >= miny; y--)
			if (primCell.data[x][y]){
				bndx[no] = x;
				bndy[no] = y;
				no++;
				if (prevY != -1){
					if (y < prevY)
						for (i = y + 1; i <= prevY - 1; i++){
							bndx[no] = x-1;
							bndy[no] = i;
							no++;
						}
					else if (y > prevY)
						for (i = prevY + 1; i <= y - 1; i++){
							bndx[no] = x;
							bndy[no] = i;
							no++;
						}
				}
				prevY = y;
				break;
			}
	return no;
}
int getLeftOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy){
	int x, y, prevY = -1, i, no = 0;
	
	for (x = minx; x <= maxx; x++)
		for (y = miny; y <= maxy; y++)
			if (primCell.data[x][y]){
				bndx[no] = x;
				bndy[no] = y;
				no++;
				if (prevY != -1){
					if (y < prevY)
						for (i = y + 1; i <= prevY - 1; i++){
							bndx[no] = x;
							bndy[no] = i;
							no++;
						}
					else if (y > prevY)
						for (i = prevY + 1; i <= y - 1; i++){
							bndx[no] = x-1;
							bndy[no] = i;
							no++;
						}
				}
				prevY = y;
				break;
			}
	return no;
}
int getTopOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy){
	int x, y, prevX = -1, i, no = 0;
	
	for (y = miny; y <= maxy; y++)
		for (x = minx; x <= maxx; x++)
			if (primCell.data[x][y]){
				bndx[no] = x;
				bndy[no] = y;
				no++;
				if (prevX != -1){
					if (x < prevX)
						for (i = x + 1; i <= prevX - 1; i++){
							bndx[no] = i;
							bndy[no] = y;
							no++;
						}
					else if (x > prevX)
						for (i = prevX + 1; i <= x - 1; i++){
							bndx[no] = i;
							bndy[no] = y-1;
							no++;
						}
				}
				prevX = x;
				break;
			}
	return no;
}
int getBottomOuterPixels(int *bndx, int *bndy, MATRIX primCell, int minx, int maxx, int miny, int maxy){
	int x, y, prevX = -1, i, no = 0;
	
	for (y = miny; y <= maxy; y++)
		for (x = maxx; x >= minx; x--)
			if (primCell.data[x][y]){
				bndx[no] = x;
				bndy[no] = y;
				no++;
				if (prevX != -1){
					if (x < prevX)
						for (i = x + 1; i <= prevX - 1; i++){
							bndx[no] = i;
							bndy[no] = y-1;
							no++;
						}
					else if (x > prevX)
						for (i = prevX + 1; i <= x - 1; i++){
							bndx[no] = i;
							bndy[no] = y;
							no++;
						}
				}
				prevX = x;				
				break;
			}
	return no;
}
void markRightOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy){
	int x, y, prevY = -1, i;
	
	for (x = minx; x <= maxx; x++)
		for (y = maxy; y >= miny; y--)
			if ((aCell->data[x][y] == 1) && (sb.data[x][y] == pid)){
				aCell->data[x][y] = 2;
				if (prevY != -1){
					if (y < prevY)
						for (i = y + 1; i <= prevY - 1; i++)
							aCell->data[x-1][i] = 2;
					else if (y > prevY)
						for (i = prevY + 1; i <= y - 1; i++)
							aCell->data[x][i] = 2;
				}
				prevY = y;
				break;
			}
}
void markLeftOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy){
	int x, y, prevY = -1, i;
	
	for (x = minx; x <= maxx; x++)
		for (y = miny; y <= maxy; y++)
			if ((aCell->data[x][y] == 1) && (sb.data[x][y] == pid)){
				aCell->data[x][y] = 2;
				if (prevY != -1){
					if (y < prevY)
						for (i = y + 1; i <= prevY - 1; i++)
							aCell->data[x][i] = 2;
					else if (y > prevY)
						for (i = prevY + 1; i <= y - 1; i++)
							aCell->data[x-1][i] = 2;
				}
				prevY = y;
				break;
			}
}
void markTopOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy){
	int x, y, prevX = -1, i;
	
	for (y = miny; y <= maxy; y++)
		for (x = minx; x <= maxx; x++)
			if ((aCell->data[x][y] == 1) && (sb.data[x][y] == pid)){
				aCell->data[x][y] = 2;
				if (prevX != -1){
					if (x < prevX)
						for (i = x + 1; i <= prevX - 1; i++)
							aCell->data[i][y] = 2;
					else if (x > prevX)
						for (i = prevX + 1; i <= x - 1; i++)
							aCell->data[i][y-1] = 2;
				}
				prevX = x;
				break;
			}
}
void markBottomOuterPixels(MATRIX *aCell, MATRIX sb, int pid, int minx, int maxx, int miny, int maxy){
	int x, y, prevX = -1, i;
	
	for (y = miny; y <= maxy; y++)
		for (x = maxx; x >= minx; x--)
			if ((aCell->data[x][y] == 1) && (sb.data[x][y] == pid)){
				aCell->data[x][y] = 2;
				if (prevX != -1){
					if (x < prevX)
						for (i = x + 1; i <= prevX - 1; i++)
							aCell->data[i][y-1] = 2;
					else if (x > prevX)
						for (i = prevX + 1; i <= x - 1; i++)
							aCell->data[i][y] = 2;
				}
				prevX = x;				
				break;
			}
}
int arePrimsNeighbor(MATRIX aCell, MATRIX p1, MATRIX p2, int id1, int id2, int minx, int maxx, int miny, int maxy){
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++){
			if (aCell.data[i][j] && (p1.data[i][j] == id1)){
				if (p2.data[i][j] == id2)													return 1;
				if ((i-1 >= minx) && aCell.data[i-1][j] && (p2.data[i-1][j] == id2))		return 1;
				if ((i+1 <= maxx) && aCell.data[i+1][j] && (p2.data[i+1][j] == id2))		return 1;
				if ((j-1 >= miny) && aCell.data[i][j-1] && (p2.data[i][j-1] == id2))		return 1;
				if ((j+1 <= maxy) && aCell.data[i][j+1] && (p2.data[i][j+1] == id2))		return 1;
			}
		}
	return 0;
}
int arePrimCellsNeighbor(MATRIX p1, MATRIX p2, int minx, int maxx, int miny, int maxy){
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++){
			if (p1.data[i][j]){
				if (p2.data[i][j])							return 1;
				if ((i-1 >= minx) && p2.data[i-1][j])		return 1;
				if ((i+1 <= maxx) && p2.data[i+1][j])		return 1;
				if ((j-1 >= miny) && p2.data[i][j-1])		return 1;
				if ((j+1 <= maxy) && p2.data[i][j+1])		return 1;
			}
		}
	return 0;
}
void computeIncrementalCentroid(int *bndx, int *bndy, int no, double *cx, double *cy, int *cno){
	int i;
	
	(*cx) *= (*cno);
	(*cy) *= (*cno);
	
	for (i = 0; i < no; i++){
		*cx += bndx[i];
		*cy += bndy[i];
	}
	(*cno) += no;
	if (*cno){
		(*cx) /= (*cno);
		(*cy) /= (*cno);
	}
}
double computeCentroidRadius(MATRIX aCell, int minx, int maxx, int miny, int maxy, double *cx, double *cy, int *cno){
	int i, j;
	double avgRadius = 0.0;
	
	*cno = 0;
	*cx = *cy = 0.0;
		
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++){
			if (aCell.data[i][j] == 2){
				(*cx) += i;
				(*cy) += j;
				(*cno)++;
			}
		}
	if (*cno){
		(*cx) /= (*cno);
		(*cy) /= (*cno);
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++){
			if (aCell.data[i][j] == 2)
				avgRadius += sqrt((i - (*cx)) * (i - (*cx)) + (j - (*cy)) * (j - (*cy)));
		}
	if (*cno)
		avgRadius /= (*cno);
	
	return avgRadius;
}
void swapPoints(int *x1, int *y1, int *x2, int *y2){
	int tmp;	
	tmp = *x1;		*x1 = *x2;		*x2 = tmp;
	tmp = *y1;		*y1 = *y2;		*y2 = tmp;
}
int getRangePoints(int x, int y, double cx, double cy, int *bndx, int *bndy, int no){
	double d = 10000,a1, a2;
	int i, ind = -1;
	
	computeAngle(x,y,cx,cy,&a1);
	for (i = 0; i < no; i++){
		computeAngle(x,y,bndx[i],bndy[i],&a2);
		if (fabs(a1 - a2) < d){
			d = fabs(a1 - a2);
			ind = i;
		}
	}
	if (d > 5)
		return -1;
	return ind;
}
void getClosestBoundaryPointsLeftRight(int *bndx1, int *bndy1, int n1, int *bndx2, int *bndy2, int n2,
									   double cx, double cy, int *p1, int *p2, int flag){
	long d, mind = -1;
	int i, j, no, start, end;
	
	if (flag == BOTTOM){	*p1 = n1 - 1;		*p2 = n2 - 1;	}
	else if (flag == TOP){	*p1 = 0;			*p2 = 0;		}
	else					terminateProgram("Flag should be TOP or BOTTOM (getClosestBoundaryPointsLeftRight)");

	mind = SQUARE(bndx1[*p1] - bndx2[*p2]) + SQUARE(bndy1[*p1] - bndy2[*p2]);
	
	for (i = 0; i < n1; i++){ 
		no = getRangePoints(bndx1[i],bndy1[i],cx,cy,bndx2,bndy2,n2);
		if (no >= 0){
			if (flag == BOTTOM){
				start = no;
				end = n2 - 1;
			}
			else{
				start = 0;
				end = no;
			}
			for (j = start; j <= end; j++){
				d = SQUARE(bndx1[i] - bndx2[j]) + SQUARE(bndy1[i] - bndy2[j]);
				if ((mind == -1) || (d < mind)){
					mind = d;
					*p1 = i;
					*p2 = j;
				}
			}
		}
	}
	
	for (i = 0; i < n2; i++){
		no = getRangePoints(bndx2[i],bndy2[i],cx,cy,bndx1,bndy1,n1);
		if (no >= 0){
			if (flag == BOTTOM){
				start = no;
				end = n1 - 1;
			}
			else{
				start = 0;
				end = no;
			}
			
			for (j = start; j <= end; j++){ 
				d = SQUARE(bndx2[i] - bndx1[j]) + SQUARE(bndy2[i] - bndy1[j]);
				if ((mind == -1) || (d < mind)){
					mind = d;
					*p1 = j;
					*p2 = i;
				}
			}
		}
	}
}
void getClosestBoundaryPointsTopBottom(int *bndx1, int *bndy1, int n1, int *bndx2, int *bndy2, int n2,
									   double cx, double cy, int *p1, int *p2, int flag){
	long d, mind = -1;
	int i, j, no, start, end;
	
	if (flag == RIGHT)		{	*p1 = n1 - 1;		*p2 = n2 - 1;	}
	else if (flag == LEFT){	*p1 = 0;			*p2 = 0;		}
	else					terminateProgram("Flag should be TOP or BOTTOM (getClosestBoundaryPointsLeftRight)");

	mind = SQUARE(bndx1[*p1] - bndx2[*p2]) + SQUARE(bndy1[*p1] - bndy2[*p2]);
	
	for (i = 0; i < n1; i++){ 
		no = getRangePoints(bndx1[i],bndy1[i],cx,cy,bndx2,bndy2,n2);
		if (no >= 0){
			if (flag == RIGHT){
				start = no;
				end = n2 - 1;
			}
			else{
				start = 0;
				end = no;
			}
			for (j = start; j <= end; j++){
				d = SQUARE(bndx1[i] - bndx2[j]) + SQUARE(bndy1[i] - bndy2[j]);
				if ((mind == -1) || (d < mind)){
					mind = d;
					*p1 = i;
					*p2 = j;
				}
			}
		}
	}
	
	for (i = 0; i < n2; i++){
		no = getRangePoints(bndx2[i],bndy2[i],cx,cy,bndx1,bndy1,n1);
		if (no >= 0){
			if (flag == RIGHT){
				start = no;
				end = n1 - 1;
			}
			else{
				start = 0;
				end = no;
			}
			
			for (j = start; j <= end; j++){ 
				d = SQUARE(bndx2[i] - bndx1[j]) + SQUARE(bndy2[i] - bndy1[j]);
				if ((mind == -1) || (d < mind)){
					mind = d;
					*p1 = j;
					*p2 = i;
				}
			}
		}
	}
}
void getClosestBoundaryPoints(int *bndx1, int *bndy1, int n1, int typ1, int *bndx2, int *bndy2, int n2, int typ2, 
							  double cx, double cy, int *p1, int *p2){
	long d, mind = -1;
	int i, j, no, start, end;
	
	if ((typ1 != RIGHT) && (typ1 != LEFT))
		terminateProgram("Typ1 should be RIGHT or LEFT (getClosestBoundaryPoints)");
	if ((typ2 != TOP) && (typ2 != BOTTOM))
		terminateProgram("Typ2 should be TOP or BOTTOM (getClosestBoundaryPoints)");
	
	if (typ2 == TOP)			*p1 = 0;
	if (typ2 == BOTTOM)			*p1 = n1 - 1;
	if (typ1 == LEFT)			*p2 = 0;
	if (typ1 == RIGHT)			*p2 = n2 - 1;
	
	mind = SQUARE(bndx1[*p1] - bndx2[*p2]) + SQUARE(bndy1[*p1] - bndy2[*p2]);

	for (i = 0; i < n1; i++){ // RL to TB
		no = getRangePoints(bndx1[i],bndy1[i],cx,cy,bndx2,bndy2,n2);
		if (no >= 0){
			if (typ1 == RIGHT){
				start = no;
				end = n2 - 1;
			}
			if (typ1 == LEFT){
				start = 0;
				end = no;
			}			
			for (j = start; j <= end; j++){
				d = SQUARE(bndx1[i] - bndx2[j]) + SQUARE(bndy1[i] - bndy2[j]);
				if ((mind == -1) || (d < mind)){
					mind = d;
					*p1 = i;
					*p2 = j;
				}
			}
		}
	}
	
	for (i = 0; i < n2; i++){ // TB to RL
		no = getRangePoints(bndx2[i],bndy2[i],cx,cy,bndx1,bndy1,n1);
		if (no >= 0){
			if (typ2 == BOTTOM){
				start = no;
				end = n1 - 1;
			}
			if (typ2 == TOP){
				start = 0;
				end = no;
			}
			
			for (j = start; j <= end; j++){ // B2R and B2L
				d = SQUARE(bndx2[i] - bndx1[j]) + SQUARE(bndy2[i] - bndy1[j]);
				if ((mind == -1) || (d < mind)){
					mind = d;
					*p1 = j;
					*p2 = i;
				}
			}
		}
	}
}
void clearNoisyOutsidePrimitives(MATRIX *aCell, MATRIX *primCell, int minx, int maxx, int miny, int maxy){
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (primCell->data[i][j])
				primCell->data[i][j] = aCell->data[i][j] = 0;
}
void fillInsideCell(MATRIX *aCell, int label, int minx, int maxx, int miny, int maxy, MATRIX res){
	int maxNo = connectedComponentsWithMask(*aCell,0,minx,maxx,miny,maxy,0,FOUR,&res);
	
	int *reached = (int *)calloc(maxNo+1,sizeof(int));
	int i, j;
	
	for (i = minx; i <= maxx; i++){
		if (res.data[i][miny] >= 0)
			reached[ res.data[i][miny] ] = 1;
		if (res.data[i][maxy] >= 0)
			reached[ res.data[i][maxy] ] = 1;
	}
	for (i = miny; i <= maxy; i++){
		if (res.data[minx][i] >= 0)
			reached[ res.data[minx][i] ] = 1;
		if (res.data[maxx][i] >= 0)
			reached[ res.data[maxx][i] ] = 1;
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if ((aCell->data[i][j] == 0) && (reached[ res.data[i][j] ] == 0))
				aCell->data[i][j] = label;
	
	free(reached);	
}
void eliminatePrimitivesBasedOnResults(MATRIX bnd, MATRIX *sb, int stype, MATRIX res, 
									   int minx, int maxx, int miny, int maxy, int d, MATRIX tmp){
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			tmp.data[i][j] = bnd.data[i][j];
	
	if(stype == BOTTOM)		eliminateBottomPrimitivesBasedOnResults(sb,minx,maxx,miny,maxy,d,tmp);
	if(stype == TOP)		eliminateTopPrimitivesBasedOnResults(sb,minx,maxx,miny,maxy,d,tmp);
	if(stype == LEFT)		eliminateLeftPrimitivesBasedOnResults(sb,minx,maxx,miny,maxy,d,tmp);
	if(stype == RIGHT)		eliminateRightPrimitivesBasedOnResults(sb,minx,maxx,miny,maxy,d,tmp);
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (res.data[i][j])
				sb->data[i][j] = 0;
}

void eliminateTopPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp){
	int i, j, k;
	
	for (j = miny; j <= maxy; j++){
		i = minx;
		while (i <= maxx){
			for (	; i <= maxx; i++)
				if (tmp.data[i][j])
					break;
			if (i <= maxx){
				for (k = 0; k < d; k++)
					if (i - k >= minx)
						tmp.data[i - k][j] = 1;
				
				for (	; i <= maxx; i++)
					if (tmp.data[i][j] == 0)
						break;
			}
			i++;
		}
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (tmp.data[i][j] && sb->data[i][j]){
				for (k = 0; i - k >= minx; k++){
					if (sb->data[i - k][j] == 0)
						break;
					sb->data[i - k][j] = 0;
				}
			}
	
}
void eliminateBottomPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp){
	int i, j, k;
	
	for (j = miny; j <= maxy; j++){
		i = maxx;
		while (i >= minx){
			for (	; i >= minx; i--)
				if (tmp.data[i][j])
					break;
			if (i >= minx){
				for (k = 0; k < d; k++)
					if (i + k <= maxx)
						tmp.data[i + k][j] = 1;
				
				for (	; i >= minx; i--)
					if (tmp.data[i][j] == 0)
						break;
			}
			i--;
		}
	}
	
	for (i = maxx; i >= minx; i--)
		for (j = miny; j <= maxy; j++)
			if (tmp.data[i][j] && sb->data[i][j]){
				for (k = 0; i + k <= maxx; k++){
					if (sb->data[i + k][j] == 0)
						break;
					sb->data[i + k][j] = 0;
				}
			}
	
}
void eliminateLeftPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp){
	int i, j, k;
	
	for (i = minx; i <= maxx; i++){
		j = miny;
		while (j <= maxy){
			for (	; j <= maxy; j++)
				if (tmp.data[i][j])
					break;
			if (j <= maxy){
				for (k = 0; k < d; k++)
					if (j - k >= miny)
						tmp.data[i][j - k] = 1;
				
				for (	; j <= maxy; j++)
					if (tmp.data[i][j] == 0)
						break;
			}
			j++;
		}
	}
	for (j = miny; j <= maxy; j++)
		for (i = minx; i <= maxx; i++)
			if (tmp.data[i][j] && sb->data[i][j]){
				for (k = 0; j - k >= miny; k++){
					if (sb->data[i][j - k] == 0)
						break;
					sb->data[i][j - k] = 0;
				}
			}
}
void eliminateRightPrimitivesBasedOnResults(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d, MATRIX tmp){
	int i, j, k;
	
	for (i = minx; i <= maxx; i++){
		j = maxy;
		while (j >= miny){
			for (	; j >= miny; j--)
				if (tmp.data[i][j])
					break;
			if (j >= miny){
				for (k = 0; k < d; k++)
					if (j + k <= maxy)
						tmp.data[i][j + k] = 1;
				
				for (	; j >= miny; j--)
					if (tmp.data[i][j] == 0)
						break;
			}
			j--;
		}
	}
	for (j = maxy; j >= miny; j--)
		for (i = minx; i <= maxx; i++)
			if (tmp.data[i][j] && sb->data[i][j]){
				for (k = 0; j + k <= maxy; k++){
					if (sb->data[i][j + k] == 0)
						break;
					sb->data[i][j + k] = 0;
				}
			}
}
void setCell(MATRIX *aCell, int minx, int maxx, int miny, int maxy){
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (aCell->data[i][j] == 2)
				aCell->data[i][j] = 1;
	
}
// It does not work properly
void completeCell(MATRIX *aCell, int x1, int y1, int x2, int y2, double cx, double cy, double avgR, 
				  int inc, int *minx, int *maxx, int *miny, int *maxy){
	int i, j, pts[1000][2], cnt, cnt2, x3, y3;
	int newMinx = *minx;
	int newMaxx = *maxx;
	int newMiny = *miny;
	int newMaxy = *maxy;
	int tmp;
	
	for (i = newMinx; i <= newMaxx; i++)
		for (j = newMiny; j <= newMaxy; j++)
			if (aCell->data[i][j] == 2)
				aCell->data[i][j] = 1;
	
	markLineBetweenTwoPoints(aCell,1,x1,y1,x2,y2);
	return;
	
	
	// BU NE KADAR ANLAMLI BILEMEDIM
	
	
	
	markLineBetweenTwoPoints(aCell,2,x1,y1,x2,y2);
	
	cnt = 1;
	if (abs(x1 - x2) > abs(y1 - y2)){
		if (x1 > x2)
			swapPoints(&x1,&y1,&x2,&y2);
		pts[0][0] = x1;
		pts[0][1] = y1;
		
		for (i = x1 + inc; i < x2; i += inc){
			for (j = newMiny; j <= newMaxy; j++)
				if (aCell->data[i][j] == 2)
					break;
			if (j > aCell->column)
				terminateProgram("Incorrect statement in complete cell");
			pts[cnt][0] = i;
			pts[cnt++][1] = j;
		}
		pts[cnt][0] = x2;
		pts[cnt++][1] = y2;
	}
	else {
		if (y1 > y2)
			swapPoints(&x1,&y1,&x2,&y2);
		pts[0][0] = x1;
		pts[0][1] = y1;
		
		for (j = y1 + inc; j < y2; j += inc){
			for (i = newMinx; i <= newMaxx; i++)
				if (aCell->data[i][j] == 2)
					break;
			if (i > aCell->row)
				terminateProgram("Incorrect statement in complete cell");
			pts[cnt][0] = i;
			pts[cnt++][1] = j;
		}
		pts[cnt][0] = x2;
		pts[cnt++][1] = y2;
	}
	if (cnt > 1000)
		terminateProgram("Increase the array size in complete cell");
	
	cnt2 = 0;
	for (i = 0; i < cnt; i++){
		x3 = computeX3(cx,pts[i][0],cy,pts[i][1],avgR);
		y3 = computeY3(cx,pts[i][0],cy,pts[i][1],x3,avgR);
		
		if (SQUARE(cx - x3) + SQUARE(cx - y3) < SQUARE(cx - pts[i][0]) + SQUARE(cy - pts[i][1]))
			continue;
		
		if (x3 < newMinx)		newMinx = x3;
		if (x3 > newMaxx)		newMaxx = x3;
		if (y3 < newMiny)		newMiny = y3;
		if (y3 > newMaxy)		newMaxy = y3;
		
		pts[cnt2][0] = x3;		
		pts[cnt2][1] = y3;	
		cnt2++;
	}
	for (i = newMinx; i <= newMaxx; i++){
		for (j = newMiny; j < *miny; j++)
			aCell->data[i][j] = 0;
		for (j = *maxy + 1; j <= newMaxy; j++)
			aCell->data[i][j] = 0;
	}
	for (j = newMiny; j <= newMaxy; j++){
		for (i = newMinx; i < *minx; i++)
			aCell->data[i][j] = 0;
		for (i = *maxx + 1; i <= newMaxx; i++)
			aCell->data[i][j] = 0;
	}
	for (i = 0; i < cnt2 - 1; i++)
		markLineBetweenTwoPoints(aCell,2,pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1]);
	markLineBetweenTwoPoints(aCell,2,pts[cnt-1][0],pts[cnt-1][1],x2,y2);
	markLineBetweenTwoPoints(aCell,2,x1,y1,pts[0][0],pts[0][1]);
	
	for (i = newMinx; i <= newMaxx; i++)
		for (j = newMiny; j <= newMaxy; j++)
			if (aCell->data[i][j])
				aCell->data[i][j] = 1;
	
	*minx = newMinx;
	*maxx = newMaxx;
	*miny = newMiny;
	*maxy = newMaxy;
}



	

