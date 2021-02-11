#include <stdio.h>
#include <stdlib.h>
#include "../util.h"
#include "../matrix.h"
#include "../imageProcessing.h"
#include "sobelPrimitives.h"
#include "primitiveOperations.h"
#include "cellCandidates.h"

/********************************************************************/
/********************************************************************/
/* Initialization of												*/
/*				1. The connected components of a mask				*/
/*				2. The Sobel responses (but NOT the primitives)		*/
/*				3. The Sobel thresholds								*/
/********************************************************************/
/********************************************************************/
CELL_IMAGE createCellImage(MATRIX gray, MATRIX mask){
	CELL_IMAGE im;
	int i;
	
	im.dx = gray.row;
	im.dy = gray.column;
	findCellImageConnectedComponents(&im,mask);
	computeSobelValues(&im,gray);	
	for (i = 0; i < 4; i++){
		im.sobelPrims[i] = allocateMatrix(im.dx,im.dy);
		initializeMatrix(&(im.sobelPrims[i]),-1);
		computeComponentSobelThresholds(&im,i);
		im.resBnd[i] = allocateMatrix(im.dx,im.dy);
		initializeMatrix(&(im.resBnd[i]),0);
		im.primCell[i] = allocateMatrix(im.dx,im.dy); 
	}
	im.adj = allocateMatrix(im.maxNo,im.maxNo);
	im.aCell = allocateMatrix(im.dx,im.dy);
	return im;
}
void freeCellImage(CELL_IMAGE im){
	int i;
	
	freeMatrix(im.conn);
	freeMatrix(im.thr);
	freeMatrix(im.bnd);
	freeMatrix(im.mask);
	freeMatrix(im.adj);
	freeMatrix(im.aCell);
	free(im.prims);
	for (i = 0; i < 4; i++){
		freeMatrix(im.sobelPrims[i]);
		freeMatrix(im.sobelValues[i]);
		freeMatrix(im.resBnd[i]);
		freeMatrix(im.primCell[i]);
	}
}
void findCellImageConnectedComponents(CELL_IMAGE *im, MATRIX mask){
	int i, j, cno;
	
	im->conn = eightConnectivity(mask);
	im->connNo = maxMatrixEntry(im->conn);
	im->bnd = allocateMatrix(im->connNo,4);
	im->thr = allocateMatrix(im->connNo,4);
	
	initializeMatrix(&(im->bnd),-1);
	for (i = 0; i < mask.row; i++)
		for (j = 0; j < mask.column; j++){
			im->conn.data[i][j] -= 1;			// so that the background pixels become -1
			cno = im->conn.data[i][j];
			if (cno == -1)
				continue;
			
			if (im->bnd.data[cno][MINX] == -1){
				im->bnd.data[cno][MINX] = i;
				im->bnd.data[cno][MAXX] = i;
				im->bnd.data[cno][MINY] = j;
				im->bnd.data[cno][MAXY] = j;
			}
			else {
				if (im->bnd.data[cno][MINX] > i)			im->bnd.data[cno][MINX] = i;
				if (im->bnd.data[cno][MAXX] < i)			im->bnd.data[cno][MAXX] = i;
				if (im->bnd.data[cno][MINY] > j)			im->bnd.data[cno][MINY] = j;
				if (im->bnd.data[cno][MAXY] < j)			im->bnd.data[cno][MAXY] = j;
			}
		}
	for (i = 0; i < im->bnd.row; i++){
		im->bnd.data[i][MINX] -= BOUNDARY_OFFSET;
		if (im->bnd.data[i][MINX] < 0)
			im->bnd.data[i][MINX] = 0;
		
		im->bnd.data[i][MAXX] += BOUNDARY_OFFSET;
		if (im->bnd.data[i][MAXX] >= im->dx)
			im->bnd.data[i][MAXX] = im->dx - 1;
		
		im->bnd.data[i][MINY] -= BOUNDARY_OFFSET;
		if (im->bnd.data[i][MINY] < 0)
			im->bnd.data[i][MINY] = 0;
		
		im->bnd.data[i][MAXY] += BOUNDARY_OFFSET;
		if (im->bnd.data[i][MAXY] >= im->dy)
			im->bnd.data[i][MAXY] = im->dy - 1;
	}
	im->primNo = 0;
	im->maxNo = 10000;
	im->prims = (PRIMITIVE *)calloc(im->maxNo,sizeof(PRIMITIVE));
	im->mask = allocateMatrix(im->dx,im->dy);
}
void computeSobelValues(CELL_IMAGE *im, MATRIX gray){
	MATRIX hx = createSobel();
	MATRIX hy = matrixTranspose(hx);
	int i, j;
	
	im->sobelValues[BOTTOM] = imfilter(gray,hx,DOUBLE,REPLICATE);
	im->sobelValues[TOP] = allocateMatrix(im->dx,im->dy);
	for (i = 0; i < im->dx; i++)
		for (j = 0; j < im->dy; j++){
			if (im->sobelValues[BOTTOM].data[i][j] > 0)
				im->sobelValues[TOP].data[i][j] = 0;
			else {
				im->sobelValues[TOP].data[i][j] = -im->sobelValues[BOTTOM].data[i][j];
				im->sobelValues[BOTTOM].data[i][j] = 0;
			}
		}
	
	im->sobelValues[RIGHT] = imfilter(gray,hy,DOUBLE,REPLICATE);
	im->sobelValues[LEFT] = allocateMatrix(im->dx,im->dy);
	for (i = 0; i < im->dx; i++)
		for (j = 0; j < im->dy; j++){
			if (im->sobelValues[RIGHT].data[i][j] > 0)
				im->sobelValues[LEFT].data[i][j] = 0;
			else {
				im->sobelValues[LEFT].data[i][j] = -im->sobelValues[RIGHT].data[i][j];
				im->sobelValues[RIGHT].data[i][j] = 0;
			}
		}

	freeMatrix(hx);
	freeMatrix(hy);
}
void computeComponentSobelThresholds(CELL_IMAGE *im, int stype){
	int **data = im->sobelValues[stype].data;
	int maxResponse = maxMatrixEntry(im->sobelValues[stype]);
	double *pvalues = (double *)calloc(maxResponse+1,sizeof(double)); 
	int i, j, k, cnt, maxP, cno;
	
	for (k = 0; k < im->connNo; k++){
		for (i = 0; i <= maxResponse; i++)
			pvalues[i] = 0.0;
		cnt = 0;
		maxP = 0;
		
		for (i = im->bnd.data[k][MINX]; i <= im->bnd.data[k][MAXX]; i++)
			for (j = im->bnd.data[k][MINY]; j <= im->bnd.data[k][MAXY]; j++)
				if (im->conn.data[i][j] == k){
					pvalues[ data[i][j] ]++;
					cnt++;
					if (data[i][j] > maxP)
						maxP = data[i][j];
				}
		for (i = 0; i <= maxP; i++)
			pvalues[i] /= cnt;
		
		im->thr.data[k][stype] = otsu(pvalues,maxP+1);
	}
	free(pvalues);
}
void thresholdSobelValues(CELL_IMAGE *im, int cid, int stype){
	int **sb = im->sobelValues[stype].data;
	int **mask = im->mask.data;
	int minx = im->bnd.data[cid][MINX];
	int maxx = im->bnd.data[cid][MAXX];
	int miny = im->bnd.data[cid][MINY];
	int maxy = im->bnd.data[cid][MAXY];
	int thr = im->thr.data[cid][stype];
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (mask[i][j] && (sb[i][j] > thr))
				im->sobelPrims[stype].data[i][j] = 1;
			else
				im->sobelPrims[stype].data[i][j] = 0;
}
void eliminateSmallSobelPrimitives(CELL_IMAGE *im, int stype, int minx, int maxx, int miny, int maxy, int wthr){
	if ((stype == RIGHT) || (stype == LEFT))
		eliminateVerticalSmallPrimitives(&(im->sobelPrims[stype]),minx,maxx,miny,maxy,wthr);
	else
		eliminateHorizontalSmallPrimitives(&(im->sobelPrims[stype]),minx,maxx,miny,maxy,wthr);
}
void findSobelBoundaries(CELL_IMAGE *im, int stype, int minx, int maxx, int miny, int maxy, int d){
	if (stype == RIGHT)
		findRightPrimitiveBoundaries(&(im->sobelPrims[RIGHT]),minx,maxx,miny,maxy,d);
	else if (stype == LEFT)
		findLeftPrimitiveBoundaries(&(im->sobelPrims[LEFT]),minx,maxx,miny,maxy,d);
	else if (stype == TOP)
		findTopPrimitiveBoundaries(&(im->sobelPrims[TOP]),minx,maxx,miny,maxy,d);
	else 
		findBottomPrimitiveBoundaries(&(im->sobelPrims[BOTTOM]),minx,maxx,miny,maxy,d);
}


void defineSobelPrimitives4ImageBoundaries(CELL_IMAGE *im, int stype){
	int i;
	
	if (stype == RIGHT){
		for (i = 0; i < im->dx; i++)
			if (im->mask.data[i][im->dy-1])
				im->sobelPrims[RIGHT].data[i][im->dy-1] = 1;
	}
	else if (stype == LEFT){
		for (i = 0; i < im->dx; i++)
			if (im->mask.data[i][0])
				im->sobelPrims[LEFT].data[i][0] = 1;
	}
	else if (stype == TOP){
		for (i = 0; i < im->dy; i++)
			if (im->mask.data[0][i])
				im->sobelPrims[TOP].data[0][i] = 1;
	}
	else if (stype == BOTTOM){
		for (i = 0; i < im->dy; i++)
			if (im->mask.data[im->dx-1][i])
				im->sobelPrims[BOTTOM].data[im->dx-1][i] = 1;
	}
}
int componentBasedEightConnectivity(MATRIX *sb, int minx, int maxx, int miny, int maxy, int nStart, MATRIX res){
	int n = connectedComponentsWithMask(*sb,1,minx,maxx,miny,maxy,nStart,FOUR,&res);
	int i, j;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			sb->data[i][j] = res.data[i][j];
	return n;
}
void extractProperties4AllPrimitives(CELL_IMAGE *im, int stype, int minx, int maxx, int miny, int maxy, int startNo, int endNo){
	int i, j, pno;
	
	for (i = startNo; i < endNo; i++){
		im->prims[i].ptype = stype;
		im->prims[i].minx = maxx + 1;
		im->prims[i].maxx = minx - 1;
		im->prims[i].miny = maxy + 1;
		im->prims[i].maxy = miny - 1;
		im->prims[i].cx = 0.0;
		im->prims[i].cy = 0.0;
		im->prims[i].area = 0;
		im->prims[i].sbAvg = 0.0;
	}
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (im->sobelPrims[stype].data[i][j] >= 0){
				pno = im->sobelPrims[stype].data[i][j];
				im->prims[pno].cx += i;
				im->prims[pno].cy += j;
				im->prims[pno].area += 1;
				im->prims[pno].sbAvg += im->sobelValues[stype].data[i][j];
				if (im->prims[pno].minx > i)			im->prims[pno].minx = i;
				if (im->prims[pno].maxx < i)			im->prims[pno].maxx = i;
				if (im->prims[pno].miny > j)			im->prims[pno].miny = j;
				if (im->prims[pno].maxy < j)			im->prims[pno].maxy = j;
			}
	for (i = startNo; i < endNo; i++){
		im->prims[i].cx /= im->prims[i].area;
		im->prims[i].cy /= im->prims[i].area;
		im->prims[i].sbAvg /= im->prims[i].area;
	}
}
void findComponentBasedPrimitives(CELL_IMAGE *im, int cid){
	MATRIX res = allocateMatrix(im->dx,im->dy);
	
	int minx = im->bnd.data[cid][MINX];
	int maxx = im->bnd.data[cid][MAXX];
	int miny = im->bnd.data[cid][MINY];
	int maxy = im->bnd.data[cid][MAXY];
	
	int nR = componentBasedEightConnectivity(&(im->sobelPrims[RIGHT]),minx,maxx,miny,maxy,0,res);
	int nL = componentBasedEightConnectivity(&(im->sobelPrims[LEFT]),minx,maxx,miny,maxy,nR,res);
	int nB = componentBasedEightConnectivity(&(im->sobelPrims[BOTTOM]),minx,maxx,miny,maxy,nL,res);
	int nT = componentBasedEightConnectivity(&(im->sobelPrims[TOP]),minx,maxx,miny,maxy,nB,res);
	
	im->primNo = nT;
	
	if (nT == 0)
		return;
	
	if (nT * 2 > im->maxNo){
		im->maxNo = nT * 2;
		free(im->prims);
		im->prims = (PRIMITIVE *)calloc(im->maxNo,sizeof(PRIMITIVE));
		freeMatrix(im->adj);
		im->adj = allocateMatrix(im->maxNo,im->maxNo);
	}
	
	extractProperties4AllPrimitives(im,RIGHT,minx,maxx,miny,maxy,0,nR);
	extractProperties4AllPrimitives(im,LEFT,minx,maxx,miny,maxy,nR,nL);
	extractProperties4AllPrimitives(im,BOTTOM,minx,maxx,miny,maxy,nL,nB);
	extractProperties4AllPrimitives(im,TOP,minx,maxx,miny,maxy,nB,nT);

	freeMatrix(res);
}
void findNeighborPixels(MATRIX *adj, int **r1, int **r2, int minx, int maxx, int miny, int maxy){
	int i, j, fnd, rid;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if (r1[i][j] >= 0){
				rid = r1[i][j];
				
				if (r2[i][j] >= 0)										adj->data[rid][r2[i][j]]   = adj->data[r2[i][j]][rid]   = 1;
				else if ((i-1 >= minx) && (r2[i-1][j] >= 0))			adj->data[rid][r2[i-1][j]] = adj->data[r2[i-1][j]][rid] = 1;
				else if ((i+1 <= maxx) && (r2[i+1][j] >= 0))			adj->data[rid][r2[i+1][j]] = adj->data[r2[i+1][j]][rid] = 1;
				else if ((j-1 >= miny) && (r2[i][j-1] >= 0))			adj->data[rid][r2[i][j-1]] = adj->data[r2[i][j-1]][rid] = 1;
				else if ((j+1 <= maxy) && (r2[i][j+1] >= 0))			adj->data[rid][r2[i][j+1]] = adj->data[r2[i][j+1]][rid] = 1;
			}
}
void findSobelNeighbors(CELL_IMAGE *im, int minx, int maxx, int miny, int maxy){ // component based minx, maxx, miny, maxy
	int **rd = im->sobelPrims[RIGHT].data;
	int **ld = im->sobelPrims[LEFT].data;
	int **bd = im->sobelPrims[BOTTOM].data;
	int **td = im->sobelPrims[TOP].data;
	int i, j;
	
	for (i = 0; i < im->primNo; i++)
		for (j = 0; j < im->primNo; j++)
			im->adj.data[i][j] = 0;
	
	findNeighborPixels(&(im->adj),rd,td,minx,maxx,miny,maxy);
	findNeighborPixels(&(im->adj),rd,bd,minx,maxx,miny,maxy);
	findNeighborPixels(&(im->adj),ld,td,minx,maxx,miny,maxy);
	findNeighborPixels(&(im->adj),ld,bd,minx,maxx,miny,maxy);
}
void find4Components(MATRIX adj, PRIMITIVE *curr, int primNo, int s1, int s2, int s3, int s4, int cells[MAXCELL][4], int *cnt){
	int i1, i2, i3, i4, j;
	int no = primNo;
	
	for (i1 = 0; i1 < no; i1++){
		if (curr[i1].ptype != s1)
			continue;
		for (i2 = 0; i2 < no; i2++){
			if ((adj.data[i1][i2] == 0) || (curr[i2].ptype != s2))
				continue;
			
			for (i3 = 0; i3 < no; i3++){
				if ((adj.data[i2][i3] == 0) || (curr[i3].ptype != s3))
					continue;
				
				for (i4 = 0; i4 < no; i4++){
					if ((adj.data[i3][i4] == 0) || (curr[i4].ptype != s4))
						continue;
					
					for (j = 0; j < *cnt; j++)
						if ((cells[j][s1] == i1) && (cells[j][s2] == i2) && (cells[j][s3] == i3) && (cells[j][s4] == i4))
							break;
					
					if (j == *cnt){
						cells[*cnt][s1] = i1;
						cells[*cnt][s2] = i2;
						cells[*cnt][s3] = i3;
						cells[*cnt][s4] = i4;
						(*cnt)++;
						
						if (*cnt == MAXCELL)
							terminateProgram("Increase the maximum cell number");
					}
				}
			}
		}
	}
}
void find3Components(MATRIX adj, PRIMITIVE *curr, int primNo, int s1, int s2, int s3, int s4, int cells[MAXCELL][4], int *cnt){
	int i1, i2, i3, i4, j;
	int no = primNo;
	
	for (i1 = 0; i1 < no; i1++){
		if (curr[i1].ptype != s1)
			continue;
		for (i2 = 0; i2 < no; i2++){
			if ((adj.data[i1][i2] == 0) || (curr[i2].ptype != s2))
				continue;
			
			for (i3 = 0; i3 < no; i3++){
				if ((adj.data[i2][i3] == 0) || (curr[i3].ptype != s3))
					continue;
				
				for (j = 0; j < *cnt; j++)
					if ((cells[j][s1] == i1) && (cells[j][s2] == i2) && (cells[j][s3] == i3))// && (cells[j][s4] == -1))
						break;
				if (j == *cnt){
					cells[*cnt][s1] = i1;
					cells[*cnt][s2] = i2;
					cells[*cnt][s3] = i3;
					cells[*cnt][s4] = -1;
					(*cnt)++;
					
					if (*cnt == MAXCELL)
						terminateProgram("Increase the maximum cell number");
				}
			}
		}
	}
}
void findCandidateCellBoundingBox(PRIMITIVE *curr, int cells[4], int *minx, int *maxx, int *miny, int *maxy){
	int i;
	
	for (i = 0; i < 4; i++)
		if (cells[i] != -1)
			break;
	
	*minx = curr[cells[i]].minx;
	*maxx = curr[cells[i]].maxx;
	*miny = curr[cells[i]].miny;
	*maxy = curr[cells[i]].maxy;
	
	for ( ; i < 4; i++){
		if (cells[i] == -1)						
			continue;
		
		if (*minx > curr[cells[i]].minx)		*minx = curr[cells[i]].minx;
		if (*maxx < curr[cells[i]].maxx)		*maxx = curr[cells[i]].maxx;
		if (*miny > curr[cells[i]].miny)		*miny = curr[cells[i]].miny;
		if (*maxy < curr[cells[i]].maxy)		*maxy = curr[cells[i]].maxy;
	}
}
int satisfyWidthConditions(CELL_IMAGE im, int minx, int maxx, int miny, int maxy, int *cellIds, int smallThr, int total){
	int rd, ld, bd, td, cnt = 0;
	
	if (cellIds[RIGHT] != -1){       
		rd = computeWidthHeight(im.primCell[RIGHT],minx,maxx,miny,maxy,HEIGHT);
		cnt += rd >= smallThr;
	}
	if (cellIds[LEFT] != -1){       
		ld = computeWidthHeight(im.primCell[LEFT],minx,maxx,miny,maxy,HEIGHT);
		cnt += ld >= smallThr;
	}
	if (cellIds[BOTTOM] != -1){
		bd = computeWidthHeight(im.primCell[BOTTOM],minx,maxx,miny,maxy,WIDTH);
		cnt += bd >= smallThr;
	}
	if (cellIds[TOP] != -1){
		td = computeWidthHeight(im.primCell[TOP],minx,maxx,miny,maxy,WIDTH);
		cnt += td >= smallThr;
	}

	if (cnt < total)
		return 0;
	
	return 1;
}
/********************************************************************/
/********************************************************************/
/********************************************************************/
/********************************************************************/
void initialize4ConnectedComponent(CELL_IMAGE *im, int cid){
	int i, j;
	
	im->primNo = 0;
	for (i = 0; i < im->dx; i++)
		for (j = 0; j < im->dy; j++)
			if (im->conn.data[i][j] == cid)
				im->mask.data[i][j] = 1;
			else
				im->mask.data[i][j] = 0;
}
void extendPrimitivesSize(CELL_IMAGE *im, int newSize){
	PRIMITIVE *tmp = im->prims;
	MATRIX tmpAdj = im->adj;
	int i, j;
	
	im->maxNo = newSize;
	im->prims = (PRIMITIVE *)calloc(newSize,sizeof(PRIMITIVE));
	for (i = 0; i < im->primNo; i++)
		im->prims[i] = tmp[i];
	free(tmp);
	
	im->adj = allocateMatrix(newSize,newSize);
	for (i = 0; i < im->primNo; i++)
		for (j = 0; j < im->primNo; j++)
			im->adj.data[i][j] = tmpAdj.data[i][j];
	freeMatrix(tmpAdj);
}
void updateSobelPrimitives(CELL_IMAGE *im, int pid, int wthr, MATRIX tmp){
	int stype	= im->prims[pid].ptype;
	int minx	= im->prims[pid].minx;
	int maxx	= im->prims[pid].maxx;
	int miny	= im->prims[pid].miny;
	int maxy	= im->prims[pid].maxy;
	int i, j, k, d, no, first;
	PRIMITIVE P;
	
	for (i = minx; i <= maxx; i++)
		for (j = miny; j <= maxy; j++)
			if ((im->aCell.data[i][j] == 1) && (im->sobelPrims[stype].data[i][j] == pid))
				im->sobelPrims[stype].data[i][j] = -1;
	
	no = connectedComponentsWithMask(im->sobelPrims[stype],pid,minx,maxx,miny,maxy,0,FOUR,&tmp); // background -1
	
	im->prims[pid].ptype = DELETED;	
	first = 0;
	for (k = 0; k < no; k++){
		P.ptype	= DELETED;
		P.minx	= maxx + 1;
		P.maxx	= minx - 1;
		P.miny	= maxy + 1;
		P.maxy	= miny - 1;
		P.cx	= 0.0;
		P.cy	= 0.0;
		P.area	= 0;
		P.sbAvg = 0.0;
		
		for (i = minx; i <= maxx; i++)
			for (j = miny; j <= maxy; j++){
				if (tmp.data[i][j] != k)
					continue;
				
				if (P.minx > i)			P.minx = i;
				if (P.maxx < i)			P.maxx = i;
				if (P.miny > j)			P.miny = j;
				if (P.maxy < j)			P.maxy = j;
				
				P.cx += i;
				P.cy += j;
				P.area += 1;
				
				P.sbAvg += im->sobelValues[stype].data[i][j];
			}
		
		if (P.area){
			P.cx /= P.area;
			P.cy /= P.area;
			P.sbAvg /= P.area;
			
			if ((stype == RIGHT) || (stype == LEFT))
				d = P.maxx - P.minx + 1;
			else
				d = P.maxy - P.miny + 1;
			
			if (d > wthr)
				P.ptype = stype;
		}
		
		if (P.ptype == DELETED){	// delete this connected component and do nothing more
			for (i = minx; i <= maxx; i++)
				for (j = miny; j <= maxy; j++)
					if (tmp.data[i][j] == k)
						im->sobelPrims[stype].data[i][j] = -1;
		}
		else {
			for ( ; first < im->primNo; first++)
				if (im->prims[first].ptype == DELETED)
					break;
			if (first == im->primNo){
				if (im->primNo == im->maxNo)
					extendPrimitivesSize(im,2 * im->maxNo);
				im->primNo += 1;
			}
			im->prims[first] = P;
			for (i = minx; i <= maxx; i++)
				for (j = miny; j <= maxy; j++)
					if (tmp.data[i][j] == k)
						im->sobelPrims[stype].data[i][j] = first;
		}
	}
}












