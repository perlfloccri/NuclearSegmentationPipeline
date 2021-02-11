
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../matrix.h"
#include "../util.h"
#include "../imageProcessing.h"
#include "sobelPrimitives.h"
#include "sobelTasks.h"
#include "watershed.h"
#include "cellCandidates.h"

void readPrims(MATRIX prims[4], char *fname){
	char str[300];

	sprintf(str,"%s_top_res",fname);		prims[TOP] = readMatrix(str);
	sprintf(str,"%s_bottom_res",fname);		prims[BOTTOM] = readMatrix(str);
	sprintf(str,"%s_left_res",fname);		prims[LEFT] = readMatrix(str);
	sprintf(str,"%s_right_res",fname);		prims[RIGHT] = readMatrix(str);
}
void freePrims(MATRIX prims[4]){
	int i;

	for (i = 0; i < 4; i++)
		freeMatrix(prims[i]);
}
void markLeftOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndy,
							 int *firstX, int *firstY, int *lastX, int *lastY){
	int i, j, val, next, first = -1, last;

	initializeMatrixPartial(outer,0,bnd.minx,bnd.maxx,bnd.miny,bnd.maxy);

	*firstX = *lastX = *firstY = *lastY = -1;
	for (i = bnd.minx; i <= bnd.maxx; i++){
		bndy[i] = -1;
		for (j = bnd.miny; j <= bnd.maxy; j++)
			if (primRes.data[i][j] == cid  &&  prim.data[i][j]){
				if (first == -1)
					first = i;
				bndy[i] = j;
				outer->data[i][j] = outerAll->data[i][j] = 1;
				last = i;
				break;
			}
	}
	if (first == -1)
		return;

	for (i = first; i < last; i++)
		if (bndy[i+1] == -1){
			for (next = i+1; next <= last && bndy[next] == -1; next++) ;
			val = (bndy[next] > bndy[i]) ? bndy[next] : bndy[i];

			for (next = i+1; next <= last && bndy[next] == -1; next++){
				bndy[next] = val;
				outer->data[next][val] = outerAll->data[next][val] = 1;
			}
			i = next - 1;
		}

	for (i = first; i < last; i++)
		if (bndy[i] != -1  &&  bndy[i+1] != -1  &&  abs(bndy[i] - bndy[i+1]) > 1){
			if (bndy[i+1] > bndy[i])
				for (j = bndy[i] + 1; j < bndy[i+1]; j++)
					outer->data[i][j] = outerAll->data[i][j] = 1;
			else
				for (j = bndy[i] - 1; j > bndy[i+1]; j--)
					outer->data[i+1][j] = outerAll->data[i+1][j] = 1;
		}
	*firstX = first;
	*firstY = bndy[first];
	*lastX = last;
	*lastY = bndy[last];
}
void markRightOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndy,
							  int *firstX, int *firstY, int *lastX, int *lastY){
	int i, j, val, next, first = -1, last;

	initializeMatrixPartial(outer,0,bnd.minx,bnd.maxx,bnd.miny,bnd.maxy);

	*firstX = *lastX = *firstY = *lastY = -1;
	for (i = bnd.minx; i <= bnd.maxx; i++){
		bndy[i] = -1;
		for (j = bnd.maxy; j >= bnd.miny; j--)
			if (primRes.data[i][j] == cid  &&  prim.data[i][j]){
				if (first == -1)
					first = i;
				bndy[i] = j;
				outer->data[i][j] = outerAll->data[i][j] = 1;
				last = i;
				break;
			}
	}
	if (first == -1)
		return;

	for (i = first; i < last; i++)
		if (bndy[i+1] == -1){
			for (next = i+1; next <= last && bndy[next] == -1; next++) ;
			val = (bndy[next] < bndy[i]) ? bndy[next] : bndy[i];

			for (next = i+1; next <= last && bndy[next] == -1; next++){
				bndy[next] = val;
				outer->data[next][val] = outerAll->data[next][val] = 1;
			}
			i = next - 1;
		}

	for (i = first; i < last; i++)
		if (bndy[i] != -1  &&  bndy[i+1] != -1  &&  abs(bndy[i] - bndy[i+1]) > 1){
			if (bndy[i+1] > bndy[i])
				for (j = bndy[i] + 1; j < bndy[i+1]; j++)
					outer->data[i+1][j] = outerAll->data[i+1][j] = 1;
			else
				for (j = bndy[i] - 1; j > bndy[i+1]; j--)
					outer->data[i][j] = outerAll->data[i][j] = 1;
		}
	*firstX = first;
	*firstY = bndy[first];
	*lastX = last;
	*lastY = bndy[last];
}
void markTopOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndx,
							int *firstX, int *firstY, int *lastX, int *lastY){
	int i, j, val, next, first = -1, last;

	initializeMatrixPartial(outer,0,bnd.minx,bnd.maxx,bnd.miny,bnd.maxy);

	*firstX = *lastX = *firstY = *lastY = -1;
	for (j = bnd.miny; j <= bnd.maxy; j++){
		bndx[j] = -1;
		for (i = bnd.minx; i <= bnd.maxx; i++)
			if (primRes.data[i][j] == cid  &&  prim.data[i][j]){
				if (first == -1)
					first = j;
				bndx[j] = i;
				outer->data[i][j] = outerAll->data[i][j] = 1;
				last = j;
				break;
			}
	}
	if (first == -1)
		return;

	for (j = first; j < last; j++)
		if (bndx[j+1] == -1){
			for (next = j+1; next <= last && bndx[next] == -1; next++) ;
			val = (bndx[next] > bndx[j]) ? bndx[next] : bndx[j];

			for (next = j+1; next <= last && bndx[next] == -1; next++){
				bndx[next] = val;
				outer->data[val][next] = outerAll->data[val][next] = 1;
			}
			j = next - 1;
		}

	for (j = first; j < last; j++)
		if (bndx[j] != -1  &&  bndx[j+1] != -1  &&  abs(bndx[j] - bndx[j+1]) > 1){
			if (bndx[j+1] > bndx[j])
				for (i = bndx[j] + 1; i < bndx[j+1]; i++)
					outer->data[i][j] = outerAll->data[i][j] = 1;
			else
				for (i = bndx[j] - 1; i > bndx[j+1]; i--)
					outer->data[i][j+1] = outerAll->data[i][j+1] = 1;
		}
	*firstY = first;
	*firstX = bndx[first];
	*lastY = last;
	*lastX = bndx[last];
}
void markBottomOuterBackground(MATRIX *outer, MATRIX *outerAll, MATRIX primRes, MATRIX prim, int cid, BNDBOX bnd, int *bndx,
							   int *firstX, int *firstY, int *lastX, int *lastY){
	int i, j, val, next, first = -1, last;

	initializeMatrixPartial(outer,0,bnd.minx,bnd.maxx,bnd.miny,bnd.maxy);

	*firstX = *lastX = *firstY = *lastY = -1;
	for (j = bnd.miny; j <= bnd.maxy; j++){
		bndx[j] = -1;
		for (i = bnd.maxx; i >= bnd.minx; i--)
			if (primRes.data[i][j] == cid  &&  prim.data[i][j]){
				if (first == -1)
					first = j;
				bndx[j] = i;
				outer->data[i][j] = outerAll->data[i][j] = 1;
				last = j;
				break;
			}
	}
	if (first == -1)
		return;

	for (j = first; j < last; j++)
		if (bndx[j+1] == -1){
			for (next = j+1; next <= last && bndx[next] == -1; next++) ;
			val = (bndx[next] < bndx[j]) ? bndx[next] : bndx[j];

			for (next = j+1; next <= last && bndx[next] == -1; next++){
				bndx[next] = val;
				outer->data[val][next] = outerAll->data[val][next] = 1;
			}
			j = next - 1;
		}

	for (j = first; j < last; j++)
		if (bndx[j] != -1  &&  bndx[j+1] != -1  &&  abs(bndx[j] - bndx[j+1]) > 1){
			if (bndx[j+1] > bndx[j])
				for (i = bndx[j] + 1; i < bndx[j+1]; i++)
					outer->data[i][j+1] = outerAll->data[i][j+1] = 1;
			else
				for (i = bndx[j] - 1; i > bndx[j+1]; i--)
					outer->data[i][j] = outerAll->data[i][j] = 1;
		}

	*firstY = first;
	*firstX = bndx[first];
	*lastY = last;
	*lastX = bndx[last];
}
int arePrimNeighbors(MATRIX primRes, int cid, MATRIX p1, MATRIX p2, BNDBOX bnd){
	int i, j, dx = p2.row, dy = p2.column;

	for (i = bnd.minx; i <= bnd.maxx; i++)
		for (j = bnd.miny; j <= bnd.maxy; j++)
			if (primRes.data[i][j] == cid  &&  p1.data[i][j]){
				if (p2.data[i][j])														return 1;
				if (i-1 >= 0  &&  primRes.data[i-1][j] == cid  &&  p2.data[i-1][j])		return 1;
				if (i+1 < dx  &&  primRes.data[i+1][j] == cid  &&  p2.data[i+1][j])		return 1;
				if (j-1 >= 0  &&  primRes.data[i][j-1] == cid  &&  p2.data[i][j-1])		return 1;
				if (j+1 < dy  &&  primRes.data[i][j+1] == cid  &&  p2.data[i][j+1])		return 1;
			}
	return 0;
}
int isEightAdjacent(int x, int y, MATRIX M){

	if (x-1 >= 0		&&  M.data[x-1][y])		return 1;
	if (x+1 < M.row		&&  M.data[x+1][y])		return 1;
	if (y-1 >= 0		&&  M.data[x][y-1])		return 1;
	if (y+1 < M.column	&&  M.data[x][y+1])		return 1;

	if (x-1 >= 0		&&  y-1 >= 0		&&  M.data[x-1][y-1])		return 1;
	if (x-1 >= 0		&&  y+1 < M.column  &&  M.data[x-1][y+1])		return 1;
	if (x+1 < M.row		&&  y-1 >= 0		&&  M.data[x+1][y-1])		return 1;
	if (x+1 < M.row		&&  y+1 < M.column  &&  M.data[x+1][y+1])		return 1;

	return 0;
}
void connectTopLeftOuter(MATRIX *outer, int topX, int topY, int lefX, int lefY){
	int smallY, largeY, smallX, largeX, k;

	if (topX < lefX  &&  topY > lefY){
		for (k = topX; k <= lefX; k++)		outer->data[k][topY] = 1;
		for (k = lefY; k <= topY; k++)		outer->data[lefX][k] = 1;
		return;
	}

	smallX = (topX < lefX) ? topX : lefX;
	largeX = (topX > lefX) ? topX : lefX;
	smallY = (topY < lefY) ? topY : lefY;
	largeY = (topY > lefY) ? topY : lefY;

	for (k = smallX; k <= largeX; k++)		outer->data[k][lefY] = 1;
	for (k = smallY; k <= largeY; k++)		outer->data[topX][k] = 1;
}
void connectTopRightOuter(MATRIX *outer, int topX, int topY, int rigX, int rigY){
	int smallY, largeY, smallX, largeX, k;

	if (topX < rigX  && topY < rigY){
		for (k = topX; k <= rigX; k++)		outer->data[k][topY] = 1;
		for (k = topY; k <= rigY; k++)		outer->data[rigX][k] = 1;
		return;
	}

	smallX = (topX < rigX) ? topX : rigX;
	largeX = (topX > rigX) ? topX : rigX;
	smallY = (topY < rigY) ? topY : rigY;
	largeY = (topY > rigY) ? topY : rigY;

	for (k = smallX; k <= largeX; k++)		outer->data[k][rigY] = 1;
	for (k = smallY; k <= largeY; k++)		outer->data[topX][k] = 1;
}
void connectBottomLeftOuter(MATRIX *outer, int botX, int botY, int lefX, int lefY){
	int smallY, largeY, smallX, largeX, k;

	if (botX > lefX  &&  botY > lefY){
		for (k = lefX; k <= botX; k++)		outer->data[k][botY] = 1;
		for (k = lefY; k <= botY; k++)		outer->data[lefX][k] = 1;
		return;
	}

	smallX = (botX < lefX) ? botX : lefX;
	largeX = (botX > lefX) ? botX : lefX;
	smallY = (botY < lefY) ? botY : lefY;
	largeY = (botY > lefY) ? botY : lefY;

	for (k = smallX; k <= largeX; k++)		outer->data[k][lefY] = 1;
	for (k = smallY; k <= largeY; k++)		outer->data[botX][k] = 1;
}
void connectBottomRightOuter(MATRIX *outer, int botX, int botY, int rigX, int rigY){
	int smallY, largeY, smallX, largeX, k;

	if (botX > rigX  &&  botY < rigY){
		for (k = rigX; k <= botX; k++)		outer->data[k][botY] = 1;
		for (k = botY; k <= rigY; k++)		outer->data[rigX][k] = 1;
		return;
	}

	smallX = (botX < rigX) ? botX : rigX;
	largeX = (botX > rigX) ? botX : rigX;
	smallY = (botY < rigY) ? botY : rigY;
	largeY = (botY > rigY) ? botY : rigY;

	for (k = smallX; k <= largeX; k++)		outer->data[k][rigY] = 1;
	for (k = smallY; k <= largeY; k++)		outer->data[botX][k] = 1;
}
void growOuterBoundaries(MATRIX *outer, MATRIX primRes, MATRIX prims[4], int d){
	MATRIX tmp = allocateMatrix(outer->row,outer->column);
	int i, j, k;

	copyMatrix(&tmp,*outer);
	for (i = 0; i < tmp.row; i++)
		for (j = 0; j < tmp.column; j++)
			if (tmp.data[i][j]){
				if (prims[TOP].data[i][j]){
					for (k = 1; k <= d; k++)
						if (i-k >= 0 && primRes.data[i-k][j] == 0)
							outer->data[i-k][j] = 1;
				}
				if (prims[BOTTOM].data[i][j]){
					for (k = 1; k <= d; k++)
						if (i+k < tmp.row && primRes.data[i+k][j] == 0)
							outer->data[i+k][j] = 1;
				}
				if (prims[LEFT].data[i][j])
					for (k = 1; k <= d; k++)
						if (j-k >= 0 && primRes.data[i][j-k] == 0)
							outer->data[i][j-k] = 1;
				if (prims[RIGHT].data[i][j])
					for (k = 1; k <= d; k++)
						if (j+k < tmp.column && primRes.data[i][j+k] == 0)
							outer->data[i][j+k] = 1;
			}
	freeMatrix(tmp);
}
MATRIX findOuterBackground(MATRIX primRes, MATRIX prims[4], int d){
	BNDBOX *bnd		= calculateBoundingBoxes(primRes,0);
	int dx			= primRes.row;
	int dy			= primRes.column;
	int *bndx		= (int *)calloc(dy,sizeof(int));
	int *bndy		= (int *)calloc(dx,sizeof(int));
	int maxNo		= maxMatrixEntry(primRes);
	MATRIX outerT	= allocateMatrix(dx,dy);
	MATRIX outerB	= allocateMatrix(dx,dy);
	MATRIX outerL	= allocateMatrix(dx,dy);
	MATRIX outerR	= allocateMatrix(dx,dy);
	MATRIX outer	= allocateMatrix(dx,dy);
	int topFX, topLX, topFY, topLY;
	int botFX, botLX, botFY, botLY;
	int rigFX, rigLX, rigFY, rigLY;
	int lefFX, lefLX, lefFY, lefLY;
	int k, topLef, topRig, botLef, botRig;

	initializeMatrix(&outer,0);
	for (k = 1; k <= maxNo; k++){
		markTopOuterBackground(&outerT,&outer,primRes,prims[TOP],k,bnd[k],bndx,&topFX,&topFY,&topLX,&topLY);
		markBottomOuterBackground(&outerB,&outer,primRes,prims[BOTTOM],k,bnd[k],bndx,&botFX,&botFY,&botLX,&botLY);
		markLeftOuterBackground(&outerL,&outer,primRes,prims[LEFT],k,bnd[k],bndy,&lefFX,&lefFY,&lefLX,&lefLY);
		markRightOuterBackground(&outerR,&outer,primRes,prims[RIGHT],k,bnd[k],bndy,&rigFX,&rigFY,&rigLX,&rigLY);

		topLef = arePrimNeighbors(primRes,k,prims[TOP],prims[LEFT],bnd[k]);
		topRig = arePrimNeighbors(primRes,k,prims[TOP],prims[RIGHT],bnd[k]);
		botLef = arePrimNeighbors(primRes,k,prims[BOTTOM],prims[LEFT],bnd[k]);
		botRig = arePrimNeighbors(primRes,k,prims[BOTTOM],prims[RIGHT],bnd[k]);

		if (topLef  &&  (!isEightAdjacent(topFX,topFY,outerL) ||  !isEightAdjacent(lefFX,lefFY,outerT)))
			connectTopLeftOuter(&outer,topFX,topFY,lefFX,lefFY);
		if (topRig  &&  (!isEightAdjacent(topLX,topLY,outerR) ||  !isEightAdjacent(rigFX,rigFY,outerT)))
			connectTopRightOuter(&outer,topLX,topLY,rigFX,rigFY);
		if (botLef  &&  (!isEightAdjacent(botFX,botFY,outerL) ||  !isEightAdjacent(lefLX,lefLY,outerB)))
			connectBottomLeftOuter(&outer,botFX,botFY,lefLX,lefLY);
		if (botRig  &&  (!isEightAdjacent(botLX,botLY,outerR) ||  !isEightAdjacent(rigLX,rigLY,outerB)))
			connectBottomRightOuter(&outer,botLX,botLY,rigLX,rigLY);
	}
	if (d > 0)
		growOuterBoundaries(&outer,primRes,prims,d);

	free(bnd);
	free(bndx);
	free(bndy);
	freeMatrix(outerT);
	freeMatrix(outerB);
	freeMatrix(outerL);
	freeMatrix(outerR);
	return outer;
}
MATRIX markCellCentroid(MATRIX primRes){
	MATRIX cent = allocateMatrix(primRes.row,primRes.column);
	int maxNo = maxMatrixEntry(primRes);
	int *cx = (int *)calloc(maxNo+1,sizeof(int));
	int *cy = (int *)calloc(maxNo+1,sizeof(int));
	int *cc = (int *)calloc(maxNo+1,sizeof(int));
	int i, j, rno;

	initializeMatrix(&cent,0);
	for (i = 0; i < primRes.row; i++)
		for (j = 0; j < primRes.column; j++)
			if (primRes.data[i][j]){
				rno = primRes.data[i][j];
				cx[rno] += i;
				cy[rno] += j;
				cc[rno]++;
			}
	for (i = 1; i <= maxNo; i++)
		if (cc[i])
			cent.data[cx[i]/cc[i]][cy[i]/cc[i]] = i;

	free(cx);
	free(cy);
	free(cc);
	return cent;
}
int initializeWatershed(MATRIX *markers, MATRIX *floodTimer, int **x, int **y, int **lastPrim, HEAP *h){
	int maxNo = maxMatrixEntry(*markers), i, j;

	*floodTimer = allocateMatrix(markers->row,markers->column);
	initializeMatrix(floodTimer,-1);

	*x = (int *)calloc(maxNo+1,sizeof(int));
	*y = (int *)calloc(maxNo+1,sizeof(int));
	*lastPrim = (int *)calloc(maxNo+1,sizeof(int));		// initialized with 0
	*h = initializeHeap(MAXHEAPSIZE,MINHEAP);

	for (i = 0; i < markers->row; i++)
		for (j = 0; j < markers->column; j++){
			if (markers->data[i][j]){
				insertHeap(h,0.0,i,j,markers->data[i][j]);
				(*x)[ markers->data[i][j] ] = i;
				(*y)[ markers->data[i][j] ] = j;
				markers->data[i][j] = 0;
			}
		}
	return maxNo;
}
void takeLastPrimMarkers(MATRIX *markers, MATRIX floodTimer, int *lastPrim){
	int i, j;

	for (i = 0; i < markers->row; i++)
		for (j = 0; j < markers->column; j++)
			if (markers->data[i][j] && (floodTimer.data[i][j] >= lastPrim[markers->data[i][j]]))
				markers->data[i][j] = 0;
}
void watershedPrimUpdate(MATRIX *markers, MATRIX primRes, MATRIX mask, MATRIX outer, int pathDist){
	HEAP h;
	HEAPDATA curr;
	MATRIX floodTimer;
	int maxNo, *x, *y, *lastPrim, cx, cy, cl, dx = mask.row, dy = mask.column;
	double d1, d2, d3, d4;

	maxNo = initializeWatershed(markers,&floodTimer,&x,&y,&lastPrim,&h);

	while (h.size > 0){
		curr = deleteHeap(&h);
		cx = curr.cx;
		cy = curr.cy;
		cl = curr.label;
		if (markers->data[cx][cy])
			continue;

		markers->data[cx][cy] = cl;
		floodTimer.data[cx][cy] = lastPrim[cl];

		if (primRes.data[cx][cy])
			lastPrim[cl]++;

		if (pathDist == YES)
			d1 = d2 = d3 = d4 = curr.key + 1;
		else {
			d1 = SQRDISTP(cx-1,cy,x[cl],y[cl]);
			d2 = SQRDISTP(cx+1,cy,x[cl],y[cl]);
			d3 = SQRDISTP(cx,cy-1,x[cl],y[cl]);
			d4 = SQRDISTP(cx,cy+1,x[cl],y[cl]);
		}

		if (cx-1 >= 0 && markers->data[cx-1][cy] == 0 && mask.data[cx-1][cy] && outer.data[cx-1][cy] == 0)		insertHeap(&h,d1,cx-1,cy,cl);
		if (cx+1 < dx && markers->data[cx+1][cy] == 0 && mask.data[cx+1][cy] && outer.data[cx+1][cy] == 0)		insertHeap(&h,d2,cx+1,cy,cl);
		if (cy-1 >= 0 && markers->data[cx][cy-1] == 0 && mask.data[cx][cy-1] && outer.data[cx][cy-1] == 0)		insertHeap(&h,d3,cx,cy-1,cl);
		if (cy+1 < dy && markers->data[cx][cy+1] == 0 && mask.data[cx][cy+1] && outer.data[cx][cy+1] == 0)		insertHeap(&h,d4,cx,cy+1,cl);
	}

	takeLastPrimMarkers(markers,floodTimer,lastPrim);

	free(x);
	free(y);
	free(lastPrim);
	freeHeap(h);
	freeMatrix(floodTimer);
}
void watershedPrimUpdateWithoutMask(MATRIX *markers, MATRIX primRes, MATRIX outer, int pathDist){
	HEAP h;
	HEAPDATA curr;
	MATRIX floodTimer;
	int maxNo, *x, *y, *lastPrim, cx, cy, cl, dx = primRes.row, dy = primRes.column;
	double d1, d2, d3, d4;

	maxNo = initializeWatershed(markers,&floodTimer,&x,&y,&lastPrim,&h);

	while (h.size > 0){
		curr = deleteHeap(&h);
		cx = curr.cx;
		cy = curr.cy;
		cl = curr.label;
		if (markers->data[cx][cy])
			continue;

		markers->data[cx][cy] = cl;
		floodTimer.data[cx][cy] = lastPrim[cl];

		if (primRes.data[cx][cy])
			lastPrim[cl]++;

		if (pathDist == YES)
			d1 = d2 = d3 = d4 = curr.key + 1;
		else {
			d1 = SQRDISTP(cx-1,cy,x[cl],y[cl]);
			d2 = SQRDISTP(cx+1,cy,x[cl],y[cl]);
			d3 = SQRDISTP(cx,cy-1,x[cl],y[cl]);
			d4 = SQRDISTP(cx,cy+1,x[cl],y[cl]);
		}

		if (cx-1 >= 0 && markers->data[cx-1][cy] == 0 && outer.data[cx-1][cy] == 0)		insertHeap(&h,d1,cx-1,cy,cl);
		if (cx+1 < dx && markers->data[cx+1][cy] == 0 && outer.data[cx+1][cy] == 0)		insertHeap(&h,d2,cx+1,cy,cl);
		if (cy-1 >= 0 && markers->data[cx][cy-1] == 0 && outer.data[cx][cy-1] == 0)		insertHeap(&h,d3,cx,cy-1,cl);
		if (cy+1 < dy && markers->data[cx][cy+1] == 0 && outer.data[cx][cy+1] == 0)		insertHeap(&h,d4,cx,cy+1,cl);
	}

	takeLastPrimMarkers(markers,floodTimer,lastPrim);

	free(x);
	free(y);
	free(lastPrim);
	freeHeap(h);
	freeMatrix(floodTimer);
}
void postProcess(MATRIX *markers, int radius){
	MATRIX se = createDiskStructuralElement(radius);
	//MATRIX res = majorityFilterFast(*markers,S_DISK,radius - 1);
	MATRIX res = majorityFilter(*markers,S_DISK,radius - 1);
	BNDBOX *bnd = calculateBoundingBoxes(res,0);
	int maxNo = maxMatrixEntry(res);
	int k;

	for (k = 1; k <= maxNo; k++)
		if (bnd[k].minx > -1)
			fillInside(&res,k,bnd[k].minx,bnd[k].maxx,bnd[k].miny,bnd[k].maxy);
	relabelComponents(&res);

	copyMatrix(markers,res);
	freeMatrix(se);
	freeMatrix(res);
	free(bnd);
}
int checkRegionGrowingArguments(int argc, char *argv[]){
	if (argc != 7){
		printf("\n\nUsage:");
		printf("\n\t0. Marker definition (1) or flooding (2)");
		printf("\n\t1. Marker primitives result file [prim]");
		printf("\n\t2. Mask file [mask]");
		//printf("\n\t3. Primitives file prefix [prim]");
		printf("\n\t3. Disk structuring element radius [5]");
		printf("\n\t4. Result file [nuclei]");
		printf("\n\t5. Produce cell nuclei (1) or cell nucleus boundaries (0)");
		printf("\n\n");
		exit(1);
	}
	return atoi(argv[4]);
}
void findCellBoundaries(MATRIX M){
	MATRIX bnd = allocateMatrix(M.row,M.column);
	int i, j;

	initializeMatrix(&bnd,0);
	for (i = 0; i < M.row; i++)
		for (j = 0; j < M.column; j++){
			if (M.data[i][j] == 0)
				continue;

			if ((i-1 >= 0		&& M.data[i-1][j] != M.data[i][j]) ||
				(i+1 < M.row	&& M.data[i+1][j] != M.data[i][j]) ||
				(j-1 >= 0		&& M.data[i][j-1] != M.data[i][j]) ||
				(j+1 < M.column	&& M.data[i][j+1] != M.data[i][j]))
				bnd.data[i][j] = M.data[i][j];
		}
	initializeMatrix(&M,0);
	for (i = 0; i < bnd.row; i++)
		for (j = 0; j < bnd.column; j++){
			if (bnd.data[i][j] == 0)
				continue;

			M.data[i][j] = bnd.data[i][j];
			if (i-1 >= 0		&& bnd.data[i-1][j] == 0)		M.data[i-1][j] = bnd.data[i][j];
			if (i+1 < M.row		&& bnd.data[i+1][j] == 0)		M.data[i+1][j] = bnd.data[i][j];
			if (j-1 >= 0		&& bnd.data[i][j-1] == 0)		M.data[i][j-1] = bnd.data[i][j];
			if (j+1 < M.column	&& bnd.data[i][j+1] == 0)		M.data[i][j+1] = bnd.data[i][j];
		}
	freeMatrix(bnd);
}
void REGION_GROWING(int argc, char *argv[]){
    
	int radius		= checkRegionGrowingArguments(argc,argv);
	MATRIX primRes	= readMatrix(argv[2]);
	MATRIX markers	= markCellCentroid(primRes);
	MATRIX prims[4], outer;

	readPrims(prims,argv[2]);
	outer = findOuterBackground(primRes,prims,0);

	watershedPrimUpdateWithoutMask(&markers,primRes,outer,YES);

	postProcess(&markers,radius);

	if (atoi(argv[6]) == 0)
		findCellBoundaries(markers);

	writeMatrixIntoFile(markers,argv[5], 1);

	freeMatrix(primRes);
	freeMatrix(outer);
	freeMatrix(markers);
	freePrims(prims);
     
}

