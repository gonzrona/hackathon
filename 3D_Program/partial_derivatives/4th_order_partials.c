#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

/*
 Find the 4th order approximation to the second derivative of the function in x,y or z direction
 Find the 4th order approximation to the first and second derivative of the function k
 ========================================================================
 general important input:
 f - the function the user wishes to find the derivative.
 Nx,Ny,Nz - the number of grid points.
 hx,hy,hz - the grid steps.
 truefxx_ext/truefyy_ext/truefzz_ext - the value of the second derivative of the function at boundary point necessary to find the 4th order approximation.
 
 k - the k_bg^2 distribution in the background media.
 A,B,C,z0,z1 - the parameter required to find the boundary condition of the first and second derivative of k(z)^2 of the form k(z) = A-B*sin(C*z)
 may be removed if the user used a different function of k(z)
 
 output:
 fxx/fyy/fzz - the 2nd order approximation to the derivative of the function in x,y or z direction
 kz,kzz - the first and second derivative of k^2 with respect to z-direction
 
 ========================================================================
 
 NOTE:
 User input are REQUIRED ONLY at the following functions:
 1) derivative_kz_4th_order
 2) derivative_kzz_4th_order
 
 Depending on the function k(z) used the user may have to manually change the boundary condition as highlighted in the code.
 Current code works only for k(z) in the form of A-B*sin(C*z) where A > B >= 0
 
 Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
 */

void derivative_xx_4th_order(int Nx, int Ny, int Nz, double hx, double complex *f, double complex *fxx, double complex *truefxx_ext){
    int i,j,l;
    double *L, *U; 
    double complex *y, *Fxx;
    Fxx = malloc(Nx*Ny*Nz * sizeof(double complex));
    L    = malloc(Nx  * sizeof(double));
    U    = malloc(Nx  * sizeof(double));
    y    = malloc(Nx  * sizeof(double complex));
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                Fxx[i+j*Nx+l*Nx*Ny] = 12.0*(f[i+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]-2.0*f[i+1+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+f[i+2+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)])/hx/hx;
            }
        }
    }
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            Fxx[j*Nx+l*Nx*Ny] = Fxx[j*Nx+l*Nx*Ny] - truefxx_ext[(j+1)+(l+1)*(Ny+2)];
            Fxx[Nx-1+j*Nx+l*Nx*Ny] = Fxx[Nx-1+j*Nx+l*Nx*Ny] - truefxx_ext[(j+1)+(l+1)*(Ny+2)+(Ny+2)*(Nz+2)];
        }
    }
    
    L[0] = 0;
    U[0] =  10.0;
    for(i=1; i<Nx; i++){
        L[i] = 1.0/U[i-1];
        U[i] = 10.0 - L[i];
    }
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            //    now solve Ly=Fxx
            y[0] = Fxx[j*Nx+l*Nx*Ny];
            for(i=1; i<Nx; i++){
                y[i] = Fxx[i+j*Nx+l*Nx*Ny] - L[i]*y[i-1];
            }
            
            //    now solve Ufxx=y
            Fxx[Nx-1+j*Nx+l*Nx*Ny] = y[Nx-1]/U[Nx-1];
            for (i=Nx-2; i>=0; i--) {
                Fxx[i+j*Nx+l*Nx*Ny] = (y[i]-Fxx[i+1+j*Nx+l*Nx*Ny])/U[i];
            }
        }
    }
    
    for (l=0; l<Nz+2; l++) {
        for (j=0; j<Ny+2; j++) {
            fxx[j*(Nx+2)+l*(Nx+2)*(Ny+2)] = truefxx_ext[j+l*(Ny+2)];
            fxx[Nx-1+j*Nx+l*(Nx+2)*(Ny+2)] = truefxx_ext[j+l*(Ny+2)+(Ny+2)*(Nz+2)];
        }
    }
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                fxx[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] = Fxx[i+j*Nx+l*Nx*Ny];
            }
        }
    }
    
    free(L);L=NULL;
    free(U);U=NULL;
    free(y);y=NULL;
    free(Fxx);Fxx = NULL;
}

void derivative_yy_4th_order(int Nx, int Ny, int Nz, double hy, double complex *f, double complex *fyy, double complex *truefyy_ext){
    int i,j,l;
    double *L, *U;
    double complex *y, *Fyy;
    Fyy = malloc(Nx*Ny*Nz * sizeof(double complex));
    L    = malloc(Ny  * sizeof(double));
    U    = malloc(Ny  * sizeof(double));
    y    = malloc(Ny  * sizeof(double complex));
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                Fyy[j+i*Ny+l*Nx*Ny] = 12.0*(f[(i+1)+j*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]-2.0*f[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+f[(i+1)+(j+2)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)])/hy/hy;
            }
        }
    }
    
    for (l=0; l<Nz; l++) {
        for (i=0; i<Nx; i++) {
            Fyy[i*Ny+l*Nx*Ny] = Fyy[i*Ny+l*Nx*Ny] - truefyy_ext[(i+1)+(l+1)*(Nx+2)];
            Fyy[Ny-1+i*Ny+l*Nx*Ny] = Fyy[Ny-1+i*Ny+l*Nx*Ny] - truefyy_ext[(i+1)+(l+1)*(Nx+2)+(Nx+2)*(Nz+2)];
        }
    }
    
    L[0] = 0;
    U[0] = 10.0;
    for(i=1; i<Ny; i++){
        L[i] = 1.0/U[i-1];
        U[i] = 10.0 - L[i];
    }
    for (l=0; l<Nz; l++) {
        for (i=0; i<Nx; i++) {
            //    now solve Ly=Fxx
            y[0] = Fyy[i*Ny+l*Nx*Ny];
            for(j=1; j<Ny; j++){
                y[j] = Fyy[j+i*Ny+l*Nx*Ny] - L[j]*y[j-1];
            }
            
            //    now solve Ufxx=y
            Fyy[Ny-1+i*Ny+l*Nx*Ny] = y[Ny-1]/U[Ny-1];
            for (j=Ny-2; j>=0; j--) {
                Fyy[j+i*Ny+l*Nx*Ny] = (y[j]-Fyy[j+1+i*Ny+l*Nx*Ny])/U[j];
            }
        }
    }
    
    for (l=0; l<Nz+2; l++) {
        for (i=0; i<Nx+2; i++) {
            fyy[i+l*(Nx+2)*(Ny+2)]=truefyy_ext[i+l*(Nx+2)];
            fyy[i+(Ny+1)*(Nx+2)+l*(Nx+2)*(Ny+2)]=truefyy_ext[i+l*(Nx+2)+(Nx+2)*(Nz+2)];
        }
    }
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                fyy[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] = Fyy[i*Ny+j+l*Nx*Ny];
            }
        }
    }
    
    free(L);L=NULL;
    free(U);U=NULL;
    free(y);y=NULL;
    free(Fyy);Fyy = NULL;
}

void derivative_zz_4th_order(int Nx, int Ny, int Nz, double hz, double complex *f, double complex *fzz, double complex *truefzz_ext){
    int i,j,l;
    double *L, *U;
    double complex *y, *Fzz;
    Fzz = malloc(Nx*Ny*Nz * sizeof(double complex));
    L    = malloc(Nz  * sizeof(double));
    U    = malloc(Nz  * sizeof(double));
    y    = malloc(Nz  * sizeof(double complex));
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                Fzz[l+i*Nz+j*Nx*Nz] = 12.0*(f[(i+1)+(j+1)*(Nx+2)+l*(Nx+2)*(Ny+2)]-2.0*f[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+f[(i+1)+(j+1)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)])/hz/hz;
            }
        }
    }

    for (j=0; j<Ny; j++) {
        for (i=0; i<Nx; i++) {
            Fzz[i*Nz+j*Nx*Nz] = Fzz[i*Nz+j*Nx*Nz] - truefzz_ext[(i+1)+(j+1)*(Nx+2)];
            Fzz[Nz-1+i*Nz+j*Nx*Nz] = Fzz[Nz-1+i*Nz+j*Nx*Nz] - truefzz_ext[(i+1)+(j+1)*(Nx+2)+(Nx+2)*(Ny+2)];
        }
    }
    
    L[0] = 0;
    U[0] = 10.0;
    for(i=1; i<Nz; i++){
        L[i] = 1.0/U[i-1];
        U[i] = 10.0 - L[i];
    }
    
    for (j=0; j<Ny; j++) {
        for (i=0; i<Nx; i++) {
            //    now solve Ly=Fxx
            y[0] = Fzz[i*Nz+j*Nx*Nz];
            for(l=1; l<Nz; l++){
                y[l] = Fzz[l+i*Nz+j*Nx*Nz] - L[l]*y[l-1];
            }
            //    now solve Ufxx=y
            Fzz[Nz-1+i*Nz+j*Nx*Nz] = y[Nz-1]/U[Nz-1];
            for (l=Nz-2; l>=0; l--) {
                Fzz[l+i*Nz+j*Nx*Nz] = (y[l]-Fzz[l+1+i*Nz+j*Nx*Nz])/U[l];
            }
        }
    }
    
    for (j=0; j<Ny+2; j++) {
        for (i=0; i<Nx+2; i++) {
            fzz[i+j*(Nx+2)]=truefzz_ext[i+j*(Nx+2)];
            fzz[i+j*(Nx+2)+(Nz+1)*(Nx+2)*(Ny+2)]=truefzz_ext[i+j*(Nx+2)+(Nx+2)*(Ny+2)];
        }
    }
    
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                fzz[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] = Fzz[l+i*Nz+j*Nx*Nz];
            }
        }
    }
    
    free(L);L=NULL;
    free(U);U=NULL;
    free(y);y=NULL;
    free(Fzz);Fzz = NULL;
}

void derivative_kz_4th_order(int Nz, double hz, double complex *k, double complex *kz,double A,double B,double C, double z0, double z1){
    int l;
    double *L, *U;
    L    = malloc(Nz  * sizeof(double));
    U    = malloc(Nz  * sizeof(double));
    
    for (l=0; l<Nz; l++) {
        kz[l] = 3.0*(k[l+2]-k[l])/hz;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////// 
    /////////// USER INPUT required : update the boundary condition as required
    /////////// Input needed: 
    /////////// The value of first derivative of k^2 with respect to z at the boundary
    /////////// In this code the boundary value was -2.0*B*C*cos(C*z0)*(A-B*sin(C*z0)) and
    /////////// - 2.0*B*C*cos(C*z1)*(A-B*sin(C*z1))
    //////////////////////////////////////////////////////////////////////////////////////////////// 

    kz[0] = kz[0] + 2.0*B*C*cos(C*z0)*(A-B*sin(C*z0));
    kz[Nz-1] = kz[Nz-1] + 2.0*B*C*cos(C*z1)*(A-B*sin(C*z1));
    
    //////////////////////////////////////////////////////////////////////////////////////////////// 
    /////////// END OF USER INPUT required 
    /////////// No further changes should be made beyond this point
    //////////////////////////////////////////////////////////////////////////////////////////////// 
    
    L[0] = 0;
    U[0] = 4.0;
    for(l=1; l<Nz; l++){
        L[l] = 1.0/U[l-1];
        U[l] = 4.0 - L[l];
    }
    
    for(l=1; l<Nz; l++){
        kz[l] = kz[l] - L[l]*kz[l-1];
    }
    
    kz[Nz-1] = kz[Nz-1]/U[Nz-1];
    for (l=Nz-2; l>=0; l--) {
        kz[l] = (kz[l]-kz[l+1])/U[l];
    }
    
    free(L);L=NULL;
    free(U);U=NULL;
}

void derivative_kzz_4th_order(int Nz, double hz, double complex *k, double complex *kzz,double A,double B,double C, double z0, double z1){
    int l;
    double *L, *U;
    L    = malloc(Nz  * sizeof(double));
    U    = malloc(Nz  * sizeof(double));
    
    for (l=0; l<Nz; l++) {
        kzz[l] = 12.0*(k[l]-2.0*k[l+1]+k[l+2])/hz/hz;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////// 
    /////////// USER INPUT required : update the boundary condition as required
    /////////// Input needed: The value of second derivative of k^2 with respect to z at the boundary
    /////////// In this code the boundary value was -2.0*B*C*C*(B*sin(C*z0)*sin(C*z0) - A*sin(C*z0) - B*cos(C*z0)*cos(C*z0)) and
    /////////// -2.0*B*C*C*(B*sin(C*z1)*sin(C*z1) - A*sin(C*z1) - B*cos(C*z1)*cos(C*z1))
    ////////////////////////////////////////////////////////////////////////////////////////////////

    kzz[0] = kzz[0] + 2.0*B*C*C*(B*sin(C*z0)*sin(C*z0) - A*sin(C*z0) - B*cos(C*z0)*cos(C*z0));
    kzz[Nz-1] = kzz[Nz-1] + 2.0*B*C*C*(B*sin(C*z1)*sin(C*z1) - A*sin(C*z1) - B*cos(C*z1)*cos(C*z1));
    
    //////////////////////////////////////////////////////////////////////////////////////////////// 
    /////////// END OF USER INPUT required : update the boundary condition as required
    //////////////////////////////////////////////////////////////////////////////////////////////// 

    L[0] = 0;
    U[0] = 10.0;

    for(l=1; l<Nz; l++){
        L[l] = 1.0/U[l-1];
        U[l] = 10.0 - L[l];
    }
    
    for(l=1; l<Nz; l++){
        kzz[l] = kzz[l] - L[l]*kzz[l-1];
    }

    kzz[Nz-1] = kzz[Nz-1]/U[Nz-1];
    for (l=Nz-2; l>=0; l--) {
        kzz[l] = (kzz[l]-kzz[l+1])/U[l];
    }
    
    free(L);L=NULL;
    free(U);U=NULL;
}
