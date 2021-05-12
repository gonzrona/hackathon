#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

/*
 Find the 2nd order approximation to the second derivative of the function either in x,y or z direction.
 ========================================================================
 general important input:
 f - the function the user wishes to find the second derivative.
 Nx,Ny,Nz - the number of grid points.
 hx,hy,hz - the grid steps.
 
 output:
 fxx/fyy/fzz - the 2nd order approximation to the second derivative of the function in x,y or z direction.
 
 ========================================================================
 
 NOTE:
 No user input required here; No changes should be made to this code.
 
 Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
 */

void derivative_xx_2nd_order(int Nx, int Ny, int Nz, double hx, double _Complex *f, double _Complex *fxx){
    int i,j,l;
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                fxx[i+j*Nx+l*Nx*Ny] = (f[i+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]-2.0*f[i+1+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+f[i+2+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)])/hx/hx;
            }
        }
    }
}

void derivative_yy_2nd_order(int Nx, int Ny, int Nz,double hy, double _Complex *f, double _Complex *fyy){
    int i,j,l;
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                fyy[i+j*Nx+l*Nx*Ny] = (f[(i+1)+j*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]-2.0*f[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+f[(i+1)+(j+2)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)])/hy/hy;
            }
        }
    }
}

void derivative_zz_2nd_order(int Nx, int Ny, int Nz, double hz, double _Complex *f, double _Complex *fzz){
    int i,j,l;
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                fzz[i+j*Nx+l*Nx*Ny] = (f[(i+1)+(j+1)*(Nx+2)+l*(Nx+2)*(Ny+2)]-2.0*f[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+f[(i+1)+(j+1)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)])/hz/hz;
            }
        }
    }
}

