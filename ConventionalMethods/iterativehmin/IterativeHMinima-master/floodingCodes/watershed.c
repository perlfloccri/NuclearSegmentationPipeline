#include "matrix.h"
#include "util.h"
#include <math.h>

/*int dn = 4;
long neighbors[][2] = {        {0,1},
                        {-1,0}       , {1,0},
                               {0,-1} };*/

int dn = 8;
long neighbors[][2] = { {-1,1}, {0,1}, {1,1},
			{-1,0}       , {1,0},
			{-1,-1},{0,-1},{1,-1} };

void findAngularPoint (int i1, int j1, int i2, int j2, double angle, int *i3, int *j3, double offset) {
	double d = sqrtf((i1-i2)*(i1-i2)+(j1-j2)*(j1-j2));
	double angle2, angle3;
	angle2 = atan(((double)(i2-i1)) / (j2-j1)) * 180.0 / PI;
    
	if (i1 < i2 && j1 > j2) {
		angle3 = -1 * angle2 + angle;
		*i3 = round(i2 + (d+offset) * sin(angle3 * PI / 180));
   		*j3 = round(j2 - (d+offset) * cos(angle3 * PI / 180));
	}
	else if (i1 < i2 && j1 < j2) {
		angle3 = angle2 + angle;
		*i3 = round(i2 + (d+offset) * sin(angle3 * PI / 180));
           	*j3 = round(j2 + (d+offset) * cos(angle3 * PI / 180));
	}
	else if (i1 > i2 && j1 < j2) {
		angle3 = -1 * angle2 + angle;
		*i3 = round(i2 - (d+offset) * sin(angle3 * PI / 180));
   		*j3 = round(j2 + (d+offset) * cos(angle3 * PI / 180));
	}
	else if (i1 > i2 && j1 > j2) {
		angle3 = angle2 + angle;
		*i3 = round(i2 - (d+offset) * sin(angle3 * PI / 180));
   		*j3 = round(j2 - (d+offset) * cos(angle3 * PI / 180));
	}
	else {
		if (i1 == i2) {
			if (j1 > j2) {
				*i3 = round(i2 - (d+offset) * sin(angle * PI / 180));
		   		*j3 = round(j2 - (d+offset) * cos(angle * PI / 180));
			}
			else {
				*i3 = round(i2 - (d+offset) * sin(angle * PI / 180));
		   		*j3 = round(j2 + (d+offset) * cos(angle * PI / 180));
			}
		}
		else {
			if (i1 > i2) {
				*i3 = round(i2 - (d+offset) * cos(angle * PI / 180));
		   		*j3 = round(j2 + (d+offset) * sin(angle * PI / 180));
			}
			else {
				*i3 = round(i2 + (d+offset) * cos(angle * PI / 180));
		   		*j3 = round(j2 + (d+offset) * sin(angle * PI / 180));
			}
		}
	}
}

void drawLine (MATRIX *mask, MATRIX *markers, int label, int i1, int j1, int i2, int j2){
	int mini, maxi, minj, maxj, it, jt, dimi, dimj, di, dj, i;
	double m, c;
    
	if (i1 < i2)	{ mini = i1;		maxi = i2; }
	else		{ mini = i2;		maxi = i1; }
	if (j1 < j2)	{ minj = j1;		maxj = j2; }
	else		{ minj = j2;		maxj = j1; }
	dimi = mask->row;	dimj = mask->column;
    
	if (i1 == i2){
		for (jt = minj; jt <= maxj; jt++) {
			if (i1 >= dimi || i1 < 0) continue;
			if (jt >= dimj || jt < 0) continue;
			if (mask->data[i1][jt] > 0 && markers->data[i1][jt] == 0)
				markers->data[i1][jt] = label;
		}
	}
	else if (j1 == j2){
		for (it = mini; it <= maxi; it++) {
			if (it >= dimi || it < 0) continue;
			if (j1 >= dimj || j1 < 0) continue;
			if (mask->data[it][j1] > 0 && markers->data[it][j1] == 0)
				markers->data[it][j1] = label;
		}
	}
	else {
		m = ((double)i1-i2) / (j1-j2);
		c = i1 - m*j1;
		for (it = mini; it < maxi; it++){
			jt = (int)round((it-c) / m);
			if (it >= dimi || it < 0)   continue;
			if (jt >= dimj || jt < 0)   continue;
			if (mask->data[it][jt] > 0) {
				markers->data[it][jt] = label;
				for (i=0; i<dn; i++) {
					di = it + neighbors[i][0];
					dj = jt + neighbors[i][1];
					if (di >= dimi || di < 0)	continue;
					if (dj >= dimj || dj < 0)	continue;
					if (mask->data[di][dj] > 0 && markers->data[di][dj] == 0) {
						markers->data[di][dj] = label;
					}
				}
			}
		}
		for (jt = minj; jt < maxj; jt++){
			it = (int)round(m*jt + c);
			if (it >= dimi || it < 0)   continue;
			if (jt >= dimj || jt < 0)   continue;
			if (mask->data[it][jt] > 0) {
				markers->data[it][jt] = label;
				for (i=0; i<dn; i++) {
					di = it + neighbors[i][0];
					dj = jt + neighbors[i][1];
					if (di >= dimi || di < 0)	continue;
					if (dj >= dimj || dj < 0)	continue;
					if (mask->data[di][dj] > 0 && markers->data[di][dj] == 0) {
						markers->data[di][dj] = label;
					}
				}
			}
		}
	}
}

int isLineInsideMask (MATRIX *mask, MATRIX *markers, int i1, int j1, int i2, int j2, int label){
	int mini, maxi, minj, maxj, it, jt, dimi, dimj, di, dj, i, k;
	double m, c;
	int step = 1;
	dimi = mask->row;	dimj = mask->column;
	if (i1 < i2)	{ mini = i1;		maxi = i2; }
	else		{ mini = i2;		maxi = i1; }
	if (j1 < j2)	{ minj = j1;		maxj = j2; }
	else		{ minj = j2;		maxj = j1; }

	if (i1 == i2){
		for (jt = minj+1; jt < maxj; jt+=step) {
			if (i1 >= dimi || i1 < 0) continue;
			if (jt >= dimj || jt < 0) continue;
			if (mask->data[i1][jt] == 0 || markers->data[i1][jt] != label) return 0;
		}
		return 1;
	}

	else if (j1 == j2){
		for (it = mini+1; it < maxi; it+=step) {
			if (it >= dimi || it < 0) continue;
			if (j1 >= dimj || j1 < 0) continue;
			if (mask->data[it][j1] == 0 || markers->data[it][j1] != label) return 0;
		}
		return 1;
	}	
	else {	
		m = ((double)i1-i2) / (j1-j2);
		c = i1 - m*j1;

		for (it = mini+1; it < maxi; it+=step){
			jt = (int)round((it-c) / m);
			if (it >= dimi || it < 0)   continue;
			if (jt >= dimj || jt < 0)   continue;
			if (mask->data[it][jt] == 0 || markers->data[it][jt] != label) return 0;
		}
		for (jt = minj+1; jt < maxj; jt+=step){
			it = (int)round(m*jt + c);
			if (it >= dimi || it < 0)   continue;
			if (jt >= dimj || jt < 0)   continue;
			if (mask->data[it][jt] == 0 || markers->data[it][jt] != label) return 0;
		}
		return 1;
	}
	
}

void findComponentCentroids (MATRIX *markers, MATRIX *sumOfPixels, MATRIX *pixelCounter) {
    int i, j, label;
    initializeMatrix (sumOfPixels, 0);
    initializeMatrix (pixelCounter, 0);
    for (i=0; i<markers->row; i++) {
        for (j=0; j<markers->column; j++) {
            label = markers->data[i][j];
            if (label) {
                sumOfPixels->data[label][0] += i;
                sumOfPixels->data[label][1] += j;
                pixelCounter->data[label][0]++;
            }
        }
    }
}

int growingCondition (MATRIX *markers, MATRIX *mask, int refI, int refJ, int i, int j, int label, double angle) {
	int i2, j2, i3, j3;
	int ta;
	if (i < 0 || j < 0 || i >= markers->row || j >= markers->column || mask->data[i][j] == 0 || markers->data[i][j] != 0 || isLineInsideMask (mask, markers, i, j, refI, refJ, label) == 0)
		return 0;
	else {
		for(ta = -1*angle; ta <= angle; ta+=1) {
			findAngularPoint (i, j, refI, refJ, ta, &i2, &j2, 0);
			if (!(i2 >= 0 && j2 >= 0 && i2 < markers->row && j2 < markers->column)) {
				findAngularPoint (i2, j2, refI, refJ, 0, &i3, &j3, 0);
				if(!(i3 >= 0 && j3 >= 0 && i3 < markers->row && j3 < markers->column) || isLineInsideMask (mask, markers, refI, refJ, i3, j3, label) == 0)
					return 0;
			}
			else if (mask->data[i2][j2] == 0 || !(markers->data[i2][j2] == 0 || markers->data[i2][j2] == label) || isLineInsideMask (mask, markers, i2, j2, i, j, label) == 0)
				return 0;
		}
		return 1;
	}
}

void growOffsetPixel(MATRIX *markers, MATRIX *mask, int centI, int centJ, int i, int j, int label, double offset) {
	int minx, maxx, miny, maxy, x, y, tx, ty, dimx, dimy, k;
	double m, c, angle;	
	int i2, j2;
	double d = sqrtf((i-centI)*(i-centI)+(j-centJ)*(j-centJ));
	findAngularPoint (i, j, centI, centJ, 0, &i2, &j2, offset);
	drawLine (mask, markers, label, centI, centJ, i2, j2);
}

void regionGrowing (MATRIX *markers, MATRIX *mask, double angle, int offset) {
	int i, j, label, i2, j2, k, centI, centJ, maxNo;
	double key;
	int offsetInd;
   	HEAP h = initializeHeap(markers->column*markers->row*dn, MINHEAP);
	HEAPDATA curr;
	MATRIX sumOfPixels, pixelCounter;
	maxNo = maxMatrixEntry(*markers);
 	sumOfPixels = allocateMatrix (maxNo+1, 2);
    	pixelCounter = allocateMatrix (maxNo+1, 1);
    	 
	findComponentCentroids (markers, &sumOfPixels, &pixelCounter);
	for (i=0; i<markers->row; i++) {
		for (j=0; j<markers->column; j++) {
			label = markers->data[i][j];
			if (label != 0) {
				centI = sumOfPixels.data[label][0] / (double)pixelCounter.data[label][0];
				centJ = sumOfPixels.data[label][1] / (double)pixelCounter.data[label][0];
				for (k=0; k<dn; k++) {
					i2 = i+neighbors[k][0];
					j2 = j+neighbors[k][1];
					if (i2 >= 0 && j2 >= 0 && i2 < markers->row && j2 < markers->column && mask->data[i2][j2] > 0 && markers->data[i2][j2] == 0) {
						key = sqrtf((centI-i2)*(centI-i2)+(centJ-j2)*(centJ-j2));
						insertHeap (&h, key, i2, j2, label);
					}
				}
			}
		}
	}
	while (h.size > 0) {
		curr = deleteHeap(&h);
		i = curr.cx;
		j = curr.cy;
		label = curr.label;

		centI = sumOfPixels.data[label][0] / (double)pixelCounter.data[label][0];
		centJ = sumOfPixels.data[label][1] / (double)pixelCounter.data[label][0];
		if (growingCondition (markers, mask, centI, centJ, i, j, label, angle)) {
			markers->data[i][j] = label;
			for (k=0; k<dn; k++) {
				i2  = i + neighbors[k][0];
				j2  = j + neighbors[k][1];
				key = sqrtf((centI-i2) * (centI-i2) + (centJ-j2) * (centJ-j2));
				insertHeap (&h, key, i2, j2, label);
			}
		}
	}

	/*grow each marker by offset pixels*/
	for (offsetInd=0; offsetInd<offset; offsetInd++) {
		for (i=0; i<markers->row; i++) {
			for (j=0; j<markers->column; j++) {
				label = markers->data[i][j];
				if (label != 0) {
					centI = sumOfPixels.data[label][0] / (double)pixelCounter.data[label][0];
					centJ = sumOfPixels.data[label][1] / (double)pixelCounter.data[label][0];
					for (k=0; k<dn; k++) {
						i2 = i+neighbors[k][0];
						j2 = j+neighbors[k][1];
						if (i2 >= 0 && j2 >= 0 && i2 < markers->row && j2 < markers->column && mask->data[i2][j2] > 0 && markers->data[i2][j2] == 0 && isLineInsideMask (mask, markers, centI, centJ, i2, j2, label) == 1) {
							key = sqrtf(neighbors[k][0]*neighbors[k][0] +neighbors[k][1]*neighbors[k][1]);
							insertHeap (&h, key, i2, j2, label);
						}
					}
				}
			}
		}
		while (h.size > 0) {
			curr = deleteHeap(&h);
			i = curr.cx;
			j = curr.cy;
			label = curr.label;
			if (curr.key <= 1)
				markers->data[i][j] = label;
		}
	}
	freeHeap(h);
    	freeMatrix(sumOfPixels);
    	freeMatrix(pixelCounter);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	MATRIX markers = convertMxArray2Matrix(prhs[0]);
	MATRIX mask    = convertMxArray2Matrix(prhs[1]);
	double angle   = mxGetScalar(prhs[2]);
	int offset = mxGetScalar(prhs[3]);
	
    regionGrowing (&markers, &mask, angle, offset);
	plhs[0] = convertMatrix2MxArray (markers);
	
    freeMatrix (mask);
	freeMatrix (markers);
}
