#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linear.h"
#include "util.h"

// computes the eigenvectors and eigenvalues of symmetric matrices
void jacobi(double **a, int n, double d[], double **v, int *nrot){
    int j, iq, ip, i;
    double tresh, theta, tau, t, sm, s, h, g, c, *b, *z;
    
    b = (double *)malloc(n * sizeof(double));
    z = (double *)malloc(n * sizeof(double));
    for (ip = 0; ip < n; ip++){
        for (iq = 0; iq < n; iq++)
            v[ip][iq] = 0.0;
        v[ip][ip] = 1.0;
    }
    for (ip = 0; ip < n; ip++){
        b[ip] = d[ip] = a[ip][ip];
        z[ip]=0.0;
    }
    *nrot = 0;
    for (i = 1; i <= 50; i++){
        //printf("iteration %d\n",i);
        sm = 0.0;
        for (ip = 0; ip < n - 1; ip++){
            for (iq = ip + 1; iq < n; iq++)
                sm += fabs(a[ip][iq]);
        }
        if (sm == 0.0){
            free(b);
            free(z);
            return;
        }
        if (i < 4)
            tresh = 0.2 * sm / (n * n);
        else
            tresh = 0.0;
        for (ip = 0; ip < n - 1; ip++){
            for (iq = ip + 1; iq < n; iq++){
                g = 100.0 * fabs(a[ip][iq]);
                if (i > 4 && (double)(fabs(d[ip]) + g) == (double)fabs(d[ip]) && 
                    (double)(fabs(d[iq]) + g) == (double)fabs(d[iq]))
                    a[ip][iq] = 0.0;
                else if (fabs(a[ip][iq]) > tresh){
                    h = d[iq] - d[ip];
                    if ((double)(fabs(h) + g) == (double)fabs(h))
                        t = (a[ip][iq]) / h;
                    else{
                        theta = 0.5 * h / (a[ip][iq]);
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0)
                            t = -t;
                    }
                    c = 1.0 / sqrt(1 + t * t);
                    s = t * c;
                    tau = s / (1.0 + c);
                    h = t * a[ip][iq];
                    z[ip] -= h;
                    z[iq] += h;
                    d[ip] -= h;
                    d[iq] += h;
                    a[ip][iq] = 0.0;
                    for (j = 0; j <= ip - 1; j++){
                        ROTATE(a,j,ip,j,iq);
                    }
                    for (j = ip + 1; j <= iq - 1; j++){
                        ROTATE(a,ip,j,j,iq);
                    }
                    for (j = iq + 1; j < n; j++){
                        ROTATE(a,ip,j,iq,j);
                    }
                    for (j = 0; j < n; j++){
                        ROTATE(v,j,ip,j,iq);
                    }
                    ++(*nrot);
                }
            }
        }
        for (ip = 0; ip < n; ip++){
            b[ip] += z[ip];
            d[ip] = b[ip];
            z[ip] = 0.0;
        }       
    }
    printf("too many iterations in routine jacobi\n");
}
void eigsrt(double d[], double **v, int n){
    int k, j, i;
    double p;
    
    for (i = 0; i < n; i++){
        p = d[k = i];
        for (j = i + 1; j < n; j++)
            if (d[j] >= p)
                p = d[k = j];
        if (k != i){
            d[k] = d[i];
            d[i] = p;
            for (j = 0; j < n; j++){
                p = v[j][i];
                v[j][i] = v[j][k];
                v[j][k] = p;
            }
        }
    }
}
void gaussj(double **a, int n, double **b, int m){
	int *indxc, *indxr, *ipiv;
	int i, icol, irow, j, k, l, ll;
	double big, dum, pivinv, temp;

	indxc = (int *)malloc(n * sizeof(int));
	indxr = (int *)malloc(n * sizeof(int));
	ipiv = (int *)malloc(n * sizeof(int));
	for (j = 0; j < n; j++)
		ipiv[j] = 0;
	for (i = 0; i < n; i++){
		big = 0.0;
		for (j = 0; j < n; j++)
			if (ipiv[j] != 1)
				for (k = 0; k < n; k++){
					if (ipiv[k] == 0){
						if (fabs(a[j][k]) >= big){
							big = fabs(a[j][k]);
							irow = j;
							icol = k;
						}
					}
					else if (ipiv[k] > 1){
						printf("\nError: gaussj: Singular matrix - 1\n");
						exit(1);
					}
				}
		++(ipiv[icol]);
		if (irow != icol){
			for (l = 0; l < n; l++)
				SWAP(a[irow][l],a[icol][l])
			for (l = 0; l < m; l++)
				SWAP(b[irow][l],b[icol][l])
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if (a[icol][icol] == 0.0){
			printf("\nError: gaussj: Singular matrix - 2\n");
			exit(1);
		}
		pivinv = 1.0 / a[icol][icol];
		a[icol][icol] = 1.0;
		for (l = 0; l < n; l++)
			a[icol][l] *= pivinv;
		for (l = 0; l < m; l++)
			b[icol][l] *= pivinv;
		for (ll = 0; ll < n; ll++)
			if (ll != icol){
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				for (l = 0; l < n; l++)
					a[ll][l] -= a[icol][l] * dum;
				for (l = 0; l < m; l++)
					b[ll][l] -= b[icol][l] * dum;
			}
	}
	for (l = n - 1; l >= 0; l--){
		if (indxr[l] != indxc[l])
			for (k = 0; k < n; k++)
				SWAP(a[k][indxr[l]],a[k][indxc[l]])
	}
	free(indxc);
	free(indxr);
	free(ipiv);
}
