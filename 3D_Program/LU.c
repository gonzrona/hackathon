#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
/*
 Calculate the LU factorization of the transformed system and the main
 diagonal in the original (untransformed) system.
 
 ========================================================================
 input:
 Nx,Ny,Nz - the number of grid points in x-,y- and z-directions.
 hx,hy,hz - the grid steps in x- and y-directions,
 k2_bg_ext    - k_bg^2 distribution in the background media.
 a - coefficient of the corners on the l-level (current level)
 b - coefficient of the points differing in y on the l-level
 c - coefficient of the points differing in x on the l-level
 d - coefficient of the center points differing on the l-level
 ap - coefficient of the corners on the (l+1)-level (top level)
 bp - coefficient of the points differing in y on the (l+1)-level
 cp - coefficient of the points differing in x on the (l+1)-level
 dp - coefficient of the center points differing on the (l+1)-level
 am - coefficient of the corners on the (l-1)-level (bottom level)
 bm - coefficient of the points differing in y on the (l-1)-level
 cm - coefficient of the points differing in x on the (l-1)-level
 dm - coefficient of the center points differing on the (l-1)-level
 
 output:
 DL       - the subdiagonal in the L matrix from the LU
 factorization of the transformed tridiagonal matrix.
 UL       - the main diagonal in the U matrix from the LU
 factorization of the transformed tridiagonal matrix.
 DM_S     - The diagonal vector of the Laplace operator which
 depends only on z variable
 ========================================================================
 
 NOTE:
 No user input required here; No changes should be made to this code.
 
 Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
 */

void LU(double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, int Nx, int Ny, int Nz, double hx, double hy, double hz, double complex *k2_bg_ext, double complex *DL,double complex *DU, double complex *DC){
  
    int l,i,j,my,mxy,mxyz,Nxy,Nxz;
    double *eigenValue_x, *eigenValue_y;
    double complex *eigenValue_3D, *eigenValue_3DP, *eigenValue_3DM;
    /*
     Introduce the eigenvalues of the matrix on every layer
     */
    Nxy = Nx*Ny;
    Nxz = Nx*Nz;
    eigenValue_x  = malloc(Nx * sizeof(double));
    eigenValue_y  = malloc(Ny * sizeof(double));
    eigenValue_3D = malloc(Nxy*Nz * sizeof(double complex));
    eigenValue_3DP = malloc(Nxy*Nz * sizeof(double complex));
    eigenValue_3DM = malloc(Nxy*Nz * sizeof(double complex));
    
    
    for( i = 0; i < Nx; i++){
        eigenValue_x[i] = 2.0*cos((i+1)*M_PI/(Nx+1));
    }

    for( j = 0; j < Ny; j++){
        eigenValue_y[j] = 2.0*cos((j+1)*M_PI/(Ny+1));
    }
    
    double eigenvalue_xy=0;
    for( j = 0; j < Ny; j++) {
        my = j*Nx;
        for( i = 0; i < Nx; i++){
            mxy = i + my;
            eigenvalue_xy = eigenValue_x[i]*eigenValue_y[j];
            for( l = 0; l < Nz; l++){
                mxyz = mxy+l*Nxy;
                eigenValue_3D[mxyz] =a[l]*eigenvalue_xy + b[l]*eigenValue_x[i] + c[l]*eigenValue_y[j]+d[l];
                eigenValue_3DP[mxyz] =ap[l]*eigenvalue_xy + bp[l]*eigenValue_x[i] + cp[l]*eigenValue_y[j]+dp[l];
                eigenValue_3DM[mxyz] =am[l]*eigenvalue_xy + bm[l]*eigenValue_x[i] + cm[l]*eigenValue_y[j]+dm[l];
            }
        }
    }
    /*
     Introduce the set of tridiagonal matrices
     */
    for( j = 0; j < Ny; j++) {
        my = j*Nxz;
        for( i = 0; i < Nx; i++){
            mxy = i*Nz + j*Nxz;
            DL[mxy] = 0.0;
            DU[mxy] = 1.0/eigenValue_3D[i + j*Nx];
            DC[mxy] = eigenValue_3DP[i+j*Nx];
        }
    }
    
    for( j = 0; j < Ny; j++) {
        my = j*Nxz;
        for( i = 0; i < Nx; i++){
            mxy = i*Nz + j*Nxz;
            for( l = 1; l < Nz; l++) {
                mxyz = l+mxy;
                DL[mxyz] = eigenValue_3DM[i+j*Nx+l*Nxy]*DU[l - 1 + i*Nz + j*Nxz];
                DU[mxyz] = 1.0/(eigenValue_3D[i+j*Nx+l*Nxy] - DC[l-1+i*Nz+j*Nxz]*DL[l + i*Nz + j*Nxz]);
                DC[mxyz] = eigenValue_3DP[i+j*Nx+l*Nxy];
            }
        }
    }
    
    
    free(eigenValue_x);
    eigenValue_x = NULL;
    free(eigenValue_y);
    eigenValue_y = NULL;
    free(eigenValue_3D);
    eigenValue_3D = NULL;
    
    return;
}



