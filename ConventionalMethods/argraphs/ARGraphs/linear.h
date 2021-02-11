#ifndef __linear_h
#define __linear_h

#define ROTATE(a,i,j,k,l) g=a[i][j]; h=a[k][l]; a[i][j]=g-s*(h+g*tau); a[k][l]=h+s*(g-h*tau);

void jacobi(double **a, int n, double d[], double **v, int *nrot);
void eigsrt(double d[], double **v, int n);
void gaussj(double **a, int n, double **b, int m);

#endif


