#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../util.h"
#include "../matrix.h"
#include "../imageProcessing.h"
#include "primitiveOperations.h"


void imfillSobelPrimitives(MATRIX *sb, int minx, int maxx, int miny, int maxy){
	MATRIX res = allocateMatrix(sb->row,sb->column);
	int maxNo, *reached, i, j;
	
	maxNo = connectedComponentsWithMask(*sb,0,minx,maxx,miny,maxy,0,FOUR,&res);
	reached = (int *)calloc(maxNo+1,sizeof(int));
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
			if ((sb->data[i][j] == 0) && (reached[ res.data[i][j] ] == 0))
				sb->data[i][j] = 1;
	free(reached);
	freeMatrix(res);
}
int connectedComponentsWithMask(MATRIX im, int cid, int minx, int maxx, int miny, int maxy, int startingLabel, int flag48, MATRIX *res){
	int *queueX = (int *)calloc((maxx - minx + 1) * (maxy - miny + 1),sizeof(int));
	int *queueY = (int *)calloc((maxx - minx + 1) * (maxy - miny + 1),sizeof(int));
	int i, j, x, y, queueStart, queueEnd;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			res->data[i][j] = -1;		// -1 means unvisited / background
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++){
			
			if (im.data[i][j] != cid)
				continue;
			if (res->data[i][j] != -1)
				continue;

			queueX[0] = i;		queueY[0] = j;
			queueStart = 0;		queueEnd = 1;
			res->data[i][j] = startingLabel;
			while (queueEnd > queueStart){
				x = queueX[queueStart];
				y = queueY[queueStart];
				queueStart++;

				if ((x-1 >= minx) && (im.data[x-1][y] == cid) && (res->data[x-1][y] == -1)){
					res->data[x-1][y] = startingLabel;
					queueX[queueEnd] = x-1;		queueY[queueEnd] = y;		queueEnd++;
				}
				if ((x+1 <= maxx) && (im.data[x+1][y] == cid) && (res->data[x+1][y] == -1)){
					res->data[x+1][y] = startingLabel;
					queueX[queueEnd] = x+1;		queueY[queueEnd] = y;		queueEnd++;
				}
				if ((y-1 >= miny) && (im.data[x][y-1] == cid) && (res->data[x][y-1] == -1)){
					res->data[x][y-1] = startingLabel;
					queueX[queueEnd] = x;		queueY[queueEnd] = y-1;		queueEnd++;
				}
				if ((y+1 <= maxy) && (im.data[x][y+1] == cid) && (res->data[x][y+1] == -1)){
					res->data[x][y+1] = startingLabel;
					queueX[queueEnd] = x;		queueY[queueEnd] = y+1;		queueEnd++;
				}
				if (flag48 == EIGHT){
					if ((x-1 >= minx) && (y-1 >= miny) && (im.data[x-1][y-1] == cid) && (res->data[x-1][y-1] == -1)){
						res->data[x-1][y-1] = startingLabel;
						queueX[queueEnd] = x-1;		queueY[queueEnd] = y-1;		queueEnd++;
					}
					if ((x-1 >= minx) && (y+1 <= maxy) && (im.data[x-1][y+1] == cid) && (res->data[x-1][y+1] == -1)){
						res->data[x-1][y+1] = startingLabel;
						queueX[queueEnd] = x-1;		queueY[queueEnd] = y+1;		queueEnd++;
					}
					if ((x+1 <= maxx) && (y-1 >= miny) && (im.data[x+1][y-1] == cid) && (res->data[x+1][y-1] == -1)){
						res->data[x+1][y-1] = startingLabel;
						queueX[queueEnd] = x+1;		queueY[queueEnd] = y-1;		queueEnd++;
					}
					if ((x+1 <= maxx) && (y+1 <= maxy) && (im.data[x+1][y+1] == cid) && (res->data[x+1][y+1] == -1)){
						res->data[x+1][y+1] = startingLabel;
						queueX[queueEnd] = x+1;		queueY[queueEnd] = y+1;		queueEnd++;
					}
				}
			}
			startingLabel++;
		}	
	free(queueX);
	free(queueY);
	return startingLabel;
}
void takeLargestComponent(MATRIX *M, int label, int minx, int maxx, int miny, int maxy, MATRIX res){
	int no = connectedComponentsWithMask(*M,label,minx,maxx,miny,maxy,0,EIGHT,&res);
	int *areas, i, j, maxInd;
	
	if (no == 1)
		return;
	
	areas = (int *)calloc(no,sizeof(int));
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (res.data[i][j] >= 0)
				areas[res.data[i][j]]++;
	maxInd = 0;
	for (i = 1; i < no; i++)
		if (areas[i] > areas[maxInd])
			maxInd = i;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (res.data[i][j] != maxInd)
				M->data[i][j] = 0;
			else
				M->data[i][j] = 1;
	
	free(areas);
}
void eliminateHorizontalSmallPrimitives(MATRIX *sb, int minx, int maxx, int miny, int maxy, int wthr){	// sb --> 0 or 1
	MATRIX res = allocateMatrix(sb->row,sb->column);
	int maxNo = connectedComponentsWithMask(*sb,1,minx,maxx,miny,maxy,0,FOUR,&res);
	int *leftY = (int *)calloc(maxNo,sizeof(int));
	int *rightY = (int *)calloc(maxNo,sizeof(int));
	int i, j;
	
	for (i = 0; i < maxNo; i++){
		leftY[i] = maxy + 1;
		rightY[i] = miny - 1;
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (res.data[i][j] >= 0){
				if (leftY[ res.data[i][j] ] > j)
					leftY[ res.data[i][j] ] = j;
				if (rightY[ res.data[i][j] ] < j)
					rightY[ res.data[i][j] ] = j;
			}
	for (i = 0; i < maxNo; i++){
		if (rightY[i] - leftY[i] <= wthr)
			rightY[i] = -1;
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if ((res.data[i][j] >= 0) && (rightY[ res.data[i][j] ] == -1))
				sb->data[i][j] = 0;
	
	free(leftY);
	free(rightY);
	freeMatrix(res);
}
void eliminateVerticalSmallPrimitives(MATRIX *sb, int minx, int maxx, int miny, int maxy, int wthr){	// sb --> 0 or 1
	MATRIX res = allocateMatrix(sb->row,sb->column);
	int maxNo = connectedComponentsWithMask(*sb,1,minx,maxx,miny,maxy,0,FOUR,&res);
	int *topX = (int *)calloc(maxNo,sizeof(int));
	int *bottomX = (int *)calloc(maxNo,sizeof(int));
	int i, j;
	
	for (i = 0; i < maxNo; i++){
		topX[i] = maxx + 1;
		bottomX[i] = minx - 1;
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (res.data[i][j] >= 0){
				if (topX[ res.data[i][j] ] > i)
					topX[ res.data[i][j] ] = i;
				if (bottomX[ res.data[i][j] ] < i)
					bottomX[ res.data[i][j] ] = i;
			}
	for (i = 0; i < maxNo; i++){
		if (bottomX[i] - topX[i] <= wthr)
			bottomX[i] = -1;
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if ((res.data[i][j] >= 0) && (bottomX[ res.data[i][j] ] == -1))
				sb->data[i][j] = 0;	
	
	free(topX);
	free(bottomX);
	freeMatrix(res);
}
void findRightPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d){
	int wflag = 1, i, j, n;

	for (i = minx; i <= maxx; i++)
		for (j = maxy; j >= miny; j--)
			if (sb->data[i][j])
				if (wflag)			wflag = 0;
				else				sb->data[i][j] = 0;
			else
				if (!wflag)			wflag = 1;
			
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			for (n = 1; n <= d; n++)
				if (sb->data[i][j] && (j - n >= 0))
					sb->data[i][j - n] = 1;
}
void findLeftPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d){
	int wflag = 1, i, j, n;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (sb->data[i][j])
				if (wflag)			wflag = 0;
				else				sb->data[i][j] = 0;
			else
				if (!wflag)			wflag = 1;
			
	for (i = minx; i <= maxx; i++)
		for (j = maxy; j >= miny; j--)
			for (n = 1; n <= d; n++)
				if (sb->data[i][j] && (j + n < sb->column))
					sb->data[i][j + n] = 1;
}
void findBottomPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d){
	int wflag = 1, i, j, n;
	
	for (i = miny; i <= maxy; i++)
		for (j = maxx; j >= minx; j--)
			if (sb->data[j][i])
				if (wflag)			wflag = 0;
				else				sb->data[j][i] = 0;
			else
				if (!wflag)			wflag = 1;

	for (i = miny; i <= maxy; i++)
		for (j = minx; j <= maxx; j++)
			for (n = 1; n <= d; n++)
				if (sb->data[j][i] && (j - n >= 0))
					sb->data[j - n][i] = 1;
}
void findTopPrimitiveBoundaries(MATRIX *sb, int minx, int maxx, int miny, int maxy, int d){
	int wflag = 1, i, j, n;
	
	for (i = miny; i <= maxy; i++)
		for (j = minx; j <= maxx; j++)
			if (sb->data[j][i])
				if (wflag)			wflag = 0;
				else				sb->data[j][i] = 0;
			else
				if (!wflag)			wflag = 1;

	for (i = miny; i <= maxy; i++)
		for (j = maxx; j >= minx; j--)
			for (n = 1; n <= d; n++)
				if (sb->data[j][i] && (j + n < sb->row))
					sb->data[j + n][i] = 1;
}
void findTopBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix){
	int x, y, first = -1, last;

	if (pid == -1){
		for (y = miny; y <= maxy; y++)
			bndPix[y] = minx;
		return;
	}	
	
	for (y = miny; y <= maxy; y++)
		for (x = minx; x <= maxx; x++)
			if (sb.data[x][y] == pid){
				bndPix[y] = x;
				if (first == -1)
					first = y;
				last = y;
				break;
			}
	for (y = miny; y < first; y++)
		bndPix[y] = bndPix[first];
	
	for (y = last+1; y <= maxy; y++)
		bndPix[y] = bndPix[last];
}
void findBottomBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix){
	int x, y, first = -1, last;
	
	if (pid == -1){
		for (y = miny; y <= maxy; y++)
			bndPix[y] = maxx;
		return;
	}
	
	for (y = miny; y <= maxy; y++)
		for (x = maxx; x >= minx; x--)
			if (sb.data[x][y] == pid){
				bndPix[y] = x;
				if (first == -1)
					first = y;
				last = y;
				break;
			}
	for (y = miny; y < first; y++)
		bndPix[y] = bndPix[first];
	
	for (y = last+1; y <= maxy; y++)
		bndPix[y] = bndPix[last];
}
void findRightBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix){
	int x, y, first = -1, last;
	if (pid == -1){
		for (x = minx; x <= maxx; x++)
			bndPix[x] = maxy;
		return;
	}
	
	for (x = minx; x <= maxx; x++)
		for (y = maxy; y >= miny; y--)
			if (sb.data[x][y] == pid){
				bndPix[x] = y;
				if (first == -1)
					first = x;
				last = x;
				break;
			}
	for (x = minx; x < first; x++)
		bndPix[x] = bndPix[first];
	
	for (x = last+1; x <= maxx; x++)
		bndPix[x] = bndPix[last];	
}
void findLeftBoundaryPixels(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *bndPix){
	int x, y, first = -1, last;

	if (pid == -1){
		for (x = minx; x <= maxx; x++)
			bndPix[x] = miny;
		return;
	}	
	
	for (x = minx; x <= maxx; x++)
		for (y = miny; y <= maxy; y++)
			if (sb.data[x][y] == pid){
                
				bndPix[x] = y;
				if (first == -1)
					first = x;
				last = x;
				break;
			}
	for (x = minx; x < first; x++)
		bndPix[x] = bndPix[first];
	
	for (x = last+1; x <= maxx; x++)
		bndPix[x] = bndPix[last];	
}
void takePrimitivesWithinBoundaries(MATRIX sb, int pid, int minx, int maxx, int miny, int maxy, int *rBnd, int *lBnd, int *bBnd, 
									int *tBnd, MATRIX *primCell, MATRIX tmp){
	int x, y;

	for (x = minx; x <= maxx; x++)
		for (y = miny; y <= maxy; y++){         
			if ((sb.data[x][y] == pid) && (x >= tBnd[y]) && (x <= bBnd[y]) && (y >= lBnd[x]) && (y <= rBnd[x]))
				primCell->data[x][y] = 1;           
			else
				primCell->data[x][y] = 0;
            }
	takeLargestComponent(primCell,1,minx,maxx,miny,maxy,tmp);
}
int computeWidthHeightWithMask(MATRIX sb, int pid, MATRIX mask, int minx, int maxx, int miny, int maxy, int flag){
	int smx = maxx + 1;
	int smy = maxy + 1;
	int lrx = minx - 1;
	int lry = miny - 1;
	int i, j, d = 0;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if ((mask.data[i][j] == 1) && (sb.data[i][j] == pid)){
				if (i < smx)		smx = i;
				if (i > lrx)		lrx = i;
				if (j < smy)		smy = j;
				if (j > lry)		lry = j;
			}
	if ((flag == HEIGHT) && (lrx >= smx))
		d = lrx - smx + 1;
	if ((flag == WIDTH) && (lry >= smy))
		d = lry - smy + 1;
	return d;
}
int computeWidthHeight(MATRIX primCell, int minx, int maxx, int miny, int maxy, int flag){
	int smx = maxx + 1;
	int smy = maxy + 1;
	int lrx = minx - 1;
	int lry = miny - 1;
	int i, j, d = 0;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (primCell.data[i][j]){
				if (i < smx)		smx = i;
				if (i > lrx)		lrx = i;
				if (j < smy)		smy = j;
				if (j > lry)		lry = j;
			}
	if ((flag == HEIGHT) && (lrx >= smx))
		d = lrx - smx + 1;
	if ((flag == WIDTH) && (lry >= smy))
		d = lry - smy + 1;
	return d;
}
int computeX3(int x1, int x2, int y1, int y2, double avgR){
	double x3, m;
	
	if (x2 - x1 < 0){
		m = (double)(y2 - y1) / (x2 - x1);
		x3 = (-1) * sqrt((avgR * avgR) / (1 + (m * m))) + x1;
	}
	else if (x2 - x1 > 0){
		m = (double)(y2 - y1) / (x2 - x1);
		x3 = sqrt((avgR * avgR) / (1 + (m * m))) + x1;
	}
	else
		x3 = x1;
	
	return (int)x3;
}
int computeY3(int x1, int x2, int y1, int y2, int x3, double avgR){
	double y3 = 0;
	double temp = (double)(avgR * avgR - ((x3 - x1) * (x3 - x1)));

	if (y2 - y1 < 0){
		if (temp < 0)
			y3 = sqrt(-temp) + y1;
		else
			y3 = -sqrt(temp) + y1;
	}
	else if (y2 - y1 > 0){
		if (temp < 0)
			y3 = sqrt(-temp) + y1;
		else
			y3 = sqrt(temp) + y1;
	}
	else
		y3 = y1;
	
	return (int)y3;
}
