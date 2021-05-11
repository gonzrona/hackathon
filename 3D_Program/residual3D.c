#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#define INDEX(i,j,l,Nx,Ny) i+j*Nx+l*Nx*Ny


/*
 Find the residual
 ========================================================================
 general important input:
 Nx,Ny,Nz     - number of grid points in x-,y- and z-directions
 hx,hy,hz     - grid steps
 SOL          - the exact solution of the equation Ax=rhs
 rhs          - the right hand side of the 3D helmholtz equation
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
 res          - residual

 
 ========================================================================
 
 NOTE:
 No user input required here; No changes should be made to this code.
 
 Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
 */

void residual3D(int Nx, int Ny, int Nz, double hx, double hy, double hz,
                double complex *SOL, double complex *res,double complex *rhs,double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, double *k2_bg_ext){
    
    int i,j,l,Nxy;
    Nxy = Nx*Ny;
    
    complex double *ext_SOL;
    ext_SOL = malloc((Nx+2)*(Ny+2)*(Nz+2) * sizeof(complex double));
    
    for(i=0; i<(Nx+2)*(Ny+2)*(Nz+2);i++){
        ext_SOL[i] = 0;
    }
    
    for(i = 0; i<Nx; i++){
        for(j = 0; j< Ny;j++){
            for(l=0; l<Nz;l++){
                ext_SOL[i+1+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] = SOL[i+j*Nx+l*Nxy];
            }
        }
    }

    double _Complex top, middle, bottom;
    for(i = 0; i<Nx; i++){
        for(j = 0; j< Ny;j++){
            for(l=0; l<Nz;l++){
                
                top = dp[l]*ext_SOL[i+1+(j+1)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)] + ap[l]*(ext_SOL[i+2+(j+2)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]+ext_SOL[i+(j+2)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]+ext_SOL[i+2+j*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]+ext_SOL[i+j*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]) + bp[l]*(ext_SOL[i+2+(j+1)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]+ext_SOL[i+(j+1)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]) + cp[l]*(ext_SOL[i+1+(j+2)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]+ext_SOL[i+1+j*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)]);
                
                middle = d[l]*ext_SOL[i+1+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] + a[l]*(ext_SOL[i+2+(j+2)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+ext_SOL[i+(j+2)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+ext_SOL[i+2+j*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+ext_SOL[i+j*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]) + b[l]*(ext_SOL[i+2+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+ext_SOL[i+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]) + c[l]*(ext_SOL[i+1+(j+2)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+ext_SOL[i+1+j*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]);
                
                bottom = dm[l]*ext_SOL[i+1+(j+1)*(Nx+2)+l*(Nx+2)*(Ny+2)] + am[l]*(ext_SOL[i+2+(j+2)*(Nx+2)+l*(Nx+2)*(Ny+2)]+ext_SOL[i+(j+2)*(Nx+2)+l*(Nx+2)*(Ny+2)]+ext_SOL[i+2+j*(Nx+2)+l*(Nx+2)*(Ny+2)]+ext_SOL[i+j*(Nx+2)+l*(Nx+2)*(Ny+2)]) + bm[l]*(ext_SOL[i+2+(j+1)*(Nx+2)+l*(Nx+2)*(Ny+2)]+ext_SOL[i+(j+1)*(Nx+2)+l*(Nx+2)*(Ny+2)]) + cm[l]*(ext_SOL[i+1+(j+2)*(Nx+2)+l*(Nx+2)*(Ny+2)]+ext_SOL[i+1+j*(Nx+2)+l*(Nx+2)*(Ny+2)]);
                
                res[i+j*Nx+l*Nxy] = (top + middle + bottom);
            }
        }
    }
    
    
    for(i=0; i<(Nx)*(Ny)*(Nz);i++){
        res[i] = res[i] - rhs[i];
    }
    
    free(ext_SOL);
    ext_SOL = NULL;
    return;
    
}

