#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
/* #define M_PI acos(-1.0) */

/*
 Find the exact solution and the right hand side of the 3D helmholtz equation based on the order user desire
 ========================================================================
 general important input:
 order - the order of approximation of your scheme: 2, 4 or 6.
 Nx,Ny,Nz - the number of grid points.
 hx,hy,hz - the grid steps.
 x0,x1,y0,y1,z0,z1 - the coordinates of the computational domain.
 k_bg_ext - k_bg_ext distribution in the background media.
 k2_bg_ext - k_bg_ext^2 distribution in the background media.
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
 A,B,C,beta,gamma - the parameter used for the test problem
 
 
 output:
 rhs - right hand side of the 3D helmholtz equation
 UE - the exact solution to the 3D helmholtz equation
 
 ========================================================================
 
 NOTE:
 User may modified the exact solution and right hand side as they desired.
 In this code we will be using UE(x,y,z) = sin(gamma*x)*sin(beta*y)*exp(-k(z)/C) as our exact solution
 and by doing so our right hand side becomes rhs(x,y,z) = -1.0*B*(2.0*A+C)*sin(C*z)*UE_ext(x,y,z)
 
 If the user decided to change the exact solution the following function would require changes as well:
 1) second_test3D_6th
 
 IMPORTANT: The user should made sure that the right hand side will satisfy the boundary condition impose on this code which
 is UE(0,y,z) = UE(x,0,z) = 0
 
 Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
 */

void derivative_xx_2nd_order(int Nx, int Ny, int Nz, double hx, double complex *f, double complex *fxx);
void derivative_yy_2nd_order(int Nx, int Ny, int Nz, double hy, double complex *f, double complex *fyy);
void derivative_zz_2nd_order(int Nx, int Ny, int Nz, double hz, double complex *f, double complex *fzz);
void derivative_xx_4th_order(int Nx, int Ny, int Nz, double hx, double complex *f, double complex *fxx, double complex *truefxx_ext);
void derivative_yy_4th_order(int Nx, int Ny, int Nz, double hy, double complex *f, double complex *fyy, double complex *truefxx_ext);
void derivative_zz_4th_order(int Nx, int Ny, int Nz, double hz, double complex *f, double complex *fzz, double complex *truefxx_ext);
void derivative_kz_4th_order(int Nz, double hz, double complex *k, double complex *kz,double A,double B,double C,double z0, double z1);


void second_test3D_2nd(int Nx, int Ny, int Nz, double complex *rhs, double complex *rhs_ext, double complex *UE_ext){

    int i,j,l;
    for( i = 0; i < Nx; i++) {
        for( j = 0; j < Ny; j++){
            for( l = 0; l < Nz; l++){
                rhs[i + j*Nx + l* Nx*Ny]  = rhs_ext[i+1 + (j+1)*(Nx+2) + (l+1)*(Nx+2)*(Ny+2)];
            }
        }
    }
    
    for( i = 0; i < Nx; i++) {
        for( j = 0; j < Ny; j++){
            rhs[i + j*Nx]  = rhs[i + j*Nx] - UE_ext[i+1 + (j+1)*(Nx+2)];
            rhs[i + j*Nx + (Nz-1)* Nx*Ny]  = rhs[i + j*Nx + (Nz-1)* Nx*Ny] - UE_ext[i+1 + (j+1)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)];
        }
    }
    
    return;
}

void second_test3D_4th(int Nx, int Ny, int Nz, double complex *rhs,double complex *rhs_ext, double complex *UE_ext, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm)
{
    
    int i,j,l;
    
    for( l = 0; l < Nz; l++) {
        for( j = 0; j < Ny; j++){
            for( i = 0; i < Nx; i++){
                rhs[i + j*Nx + l*Nx*Ny] = rhs_ext[i+1 + (j+1)*(Nx+2) + (l+1)*(Nx+2)*(Ny+2)];
            }
        }
    }
    
    for( l = 0; l < Nz; l++) {
        for( j = 0; j < Ny; j++){
            for( i = 0; i < Nx; i++){
                rhs[i + j*Nx + l*Nx*Ny] = rhs[i + j*Nx + l*Nx*Ny] + (1.0/12.0)*(rhs_ext[i+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+ rhs_ext[i+2+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] + rhs_ext[i+1+(j)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] + rhs_ext[i+1+(j+2)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)] + rhs_ext[i+1+(j+1)*(Nx+2)+l*(Nx+2)*(Ny+2)] + rhs_ext[i+1+(j+1)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)] -6*rhs_ext[i+1+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]);
            }
        }
    }
    
    for( i = 0; i < Nx; i++) {
        for( j = 0; j < Ny; j++){
            rhs[i + j*Nx]  = rhs[i + j*Nx] - am[0]*(UE_ext[i + j*(Nx+2)]+UE_ext[i+2 + j*(Nx+2)] + UE_ext[i + (j+2)*(Nx+2)] + UE_ext[i+2 + (j+2)*(Nx+2)]) - bm[0]*(UE_ext[i + (j+1)*(Nx+2)]+UE_ext[i+2 + (j+1)*(Nx+2)]) - cm[0]*(UE_ext[i+1 + j*(Nx+2)] + UE_ext[i+1 + (j+2)*(Nx+2)]) - dm[0]*UE_ext[i+1 + (j+1)*(Nx+2)];
            
            
            rhs[i + j*Nx + (Nz-1)*Nx*Ny]  = rhs[i + j*Nx + (Nz-1)*Nx*Ny] - ap[Nz-1]*(UE_ext[i + j*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)] + UE_ext[i+2 + j*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)] + UE_ext[i + (j+2)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)] + UE_ext[i+2 + (j+2)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]) - bp[Nz-1]*(UE_ext[i + (j+1)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]+UE_ext[i+2 + (j+1)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]) - cp[Nz-1]*(UE_ext[i+1 + j*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]+UE_ext[i+1 + (j+2)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]) - dp[Nz-1]*UE_ext[i+1 + (j+1)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)];
        }
    }
    return;
    
}

void second_test3D_6th(int Nx, int Ny, int Nz, double hx, double hy, double hz, double x0, double y0, double z0, double z1,double complex *k_bg_ext, double complex *k2_bg_ext, double complex *rhs, double complex *rhs_ext, double complex *UE_ext,double A, double B, double C, double beta, double gamma, double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm)
{
    int i,j,l;
  
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////// USER INPUT required : If the user change the exact solution they need to update the 
    ///////////                       the boundary value of second derivative of right hand side accordingly
    /////////// Input needed: 
    /////////// truefxx_ext - the boundary value of second derivative of right hand side in x-direction
    /////////// truefyy_ext - the boundary value of second derivative of right hand side in y-direction
    /////////// truefzz_ext - the boundary value of second derivative of right hand side in z-direction
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    

    double complex *truefxx_ext;
    truefxx_ext = malloc(2*(Ny+2)*(Nz+2) * sizeof(double complex));
    double complex *truefyy_ext;
    truefyy_ext = malloc(2*(Nx+2)*(Nz+2) * sizeof(double complex));
    double complex *truefzz_ext;
    truefzz_ext = malloc(2*(Nx+2)*(Ny+2) * sizeof(double complex));
  
    for (l=0; l<Nz+2; l++) {
        for (j=0; j<Ny+2; j++) {
            truefxx_ext[j+l*(Ny+2)] = -1.0*gamma*gamma*rhs_ext[j*(Nx+2) + l*(Nx+2)*(Ny+2)];
            truefxx_ext[(Ny+2)*(Nz+2)+j+l*(Ny+2)] = -1.0*gamma*gamma*rhs_ext[Nx+1 + j*(Nx+2) + l*(Nx+2)*(Ny+2)];
        }
    }
    
    for (l=0; l<Nz+2; l++) {
        for (i=0; i<Nx+2; i++) {
            truefyy_ext[i+l*(Nx+2)] = -1.0*beta*beta*rhs_ext[i + l*(Nx+2)*(Ny+2)];
            truefyy_ext[(Nx+2)*(Nz+2)+i+l*(Nx+2)] = -1.0*beta*beta*rhs_ext[i + (Ny+1)*(Nx+2) + l*(Nx+2)*(Ny+2)];
        }
    }

    for (j=0; j<Ny+2; j++) {
        for (i=0; i<Nx+2; i++) {
            truefzz_ext[i+j*(Nx+2)] = hz*hz*B*(2.0*A+C)*UE_ext[i + j*(Nx+2)]*(B*C*sin(C*z0)*sin(C*z0) + sin(C*z0)*(C*C-B*B*cos(C*z0)*cos(C*z0)) - 2.0*B*C*cos(C*z0)*cos(C*z0));
            truefzz_ext[(Nx+2)*(Ny+2)+i+j*(Nx+2)] = hz*hz*B*(2.0*A+C)*UE_ext[i + j*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]*(B*C*sin(C*z1)*sin(C*z1) + sin(C*z1)*(C*C-B*B*cos(C*z1)*cos(C*z1)) - 2.0*B*C*cos(C*z1)*cos(C*z1));
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////// END OF USER INPUT required 
    /////////// No further changes should be made beyond this point
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    double complex *fxx_ext, *fyy_ext, *fzz_ext;
    fxx_ext = malloc((Nx+2)*(Ny+2)*(Nz+2) * sizeof(double complex));
    fyy_ext = malloc((Nx+2)*(Ny+2)*(Nz+2) * sizeof(double complex));
    fzz_ext = malloc((Nx+2)*(Ny+2)*(Nz+2) * sizeof(double complex));
    
    derivative_xx_4th_order(Nx,Ny,Nz,hx,rhs_ext,fxx_ext,truefxx_ext);
    derivative_yy_4th_order(Nx,Ny,Nz,hy,rhs_ext,fyy_ext,truefyy_ext);
    derivative_zz_4th_order(Nx,Ny,Nz,hz,rhs_ext,fzz_ext,truefzz_ext);
    
    double complex *fxxxx, *fyyyy, *fzzzz, *fxxyy, *fxxzz, *fyyzz;
    fxxxx = malloc(Nx*Ny*Nz * sizeof(double complex));
    fyyyy = malloc(Nx*Ny*Nz * sizeof(double complex));
    fzzzz = malloc(Nx*Ny*Nz * sizeof(double complex));
    
    fxxyy = malloc(Nx*Ny*Nz * sizeof(double complex));
    fxxzz = malloc(Nx*Ny*Nz * sizeof(double complex));
    fyyzz = malloc(Nx*Ny*Nz * sizeof(double complex));
    
    derivative_xx_2nd_order(Nx, Ny, Nz, hx, fxx_ext, fxxxx);
    derivative_yy_2nd_order(Nx, Ny, Nz, hy, fyy_ext, fyyyy);
    derivative_zz_2nd_order(Nx, Ny, Nz, hz, fzz_ext, fzzzz);

    derivative_yy_2nd_order(Nx, Ny, Nz, hy, fxx_ext, fxxyy);
    derivative_zz_2nd_order(Nx, Ny, Nz, hz, fxx_ext, fxxzz);
    derivative_zz_2nd_order(Nx, Ny, Nz, hz, fyy_ext, fyyzz);
    
    double complex *k_prime = malloc(Nz * sizeof(double complex));

    derivative_kz_4th_order(Nz, hz, k2_bg_ext, k_prime,A,B,C,z0,z1);
    
    
    double first, second, third, fourth, fifth;
    for (l=0; l<Nz; l++) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                first = (1.0-k2_bg_ext[l+1]*hz*hz/20.0)*rhs_ext[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)];
                
                second = (hz*hz/12.0)*(fxx_ext[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+fyy_ext[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]+fzz_ext[(i+1)+(j+1)*(Nx+2)+(l+1)*(Nx+2)*(Ny+2)]);
                
                third = (hz*hz*hz*hz/360.0)*(fxxxx[i+j*Nx+l*Nx*Ny]+fyyyy[i+j*Nx+l*Nx*Ny]+fzzzz[i+j*Nx+l*Nx*Ny]);
                
                fourth = (hz*hz*hz*hz/90.0)*(fxxyy[i+j*Nx+l*Nx*Ny]+fxxzz[i+j*Nx+l*Nx*Ny]+fyyzz[i+j*Nx+l*Nx*Ny]);
                
                fifth = (hz*hz*hz/120.0)*k_prime[l]*(rhs_ext[(i+1)+(j+1)*(Nx+2)+(l+2)*(Nx+2)*(Ny+2)] - rhs_ext[(i+1)+(j+1)*(Nx+2)+l*(Nx+2)*(Ny+2)]);
                
                rhs[i+j*Nx+l*Nx*Ny] = (first+second+third+fourth+fifth);
            }
        }
    }
    for( i = 0; i < Nx; i++) {
        for( j = 0; j < Ny; j++){
            rhs[i + j*Nx]  = rhs[i + j*Nx] - am[0]*(UE_ext[i + j*(Nx+2)]+UE_ext[i+2 + j*(Nx+2)] + UE_ext[i + (j+2)*(Nx+2)] + UE_ext[i+2 + (j+2)*(Nx+2)]) - bm[0]*(UE_ext[i + (j+1)*(Nx+2)]+UE_ext[i+2 + (j+1)*(Nx+2)]) - cm[0]*(UE_ext[i+1 + j*(Nx+2)] + UE_ext[i+1 + (j+2)*(Nx+2)]) - dm[0]*UE_ext[i+1 + (j+1)*(Nx+2)];
            
            
            rhs[i + j*Nx + (Nz-1)*Nx*Ny]  = rhs[i + j*Nx + (Nz-1)*Nx*Ny] - ap[Nz-1]*(UE_ext[i + j*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)] + UE_ext[i+2 + j*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)] + UE_ext[i + (j+2)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)] + UE_ext[i+2 + (j+2)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]) - bp[Nz-1]*(UE_ext[i + (j+1)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]+UE_ext[i+2 + (j+1)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]) - cp[Nz-1]*(UE_ext[i+1 + j*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]+UE_ext[i+1 + (j+2)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)]) - dp[Nz-1]*UE_ext[i+1 + (j+1)*(Nx+2) + (Nz+1)*(Nx+2)*(Ny+2)];
        }
    }
    
    free(k_prime); k_prime = NULL;
    
    free(truefxx_ext);truefxx_ext = NULL;
    free(truefyy_ext);truefyy_ext = NULL;
    free(truefzz_ext);truefzz_ext = NULL;
    
    free(fxx_ext);fxx_ext = NULL;
    free(fyy_ext);fyy_ext = NULL;
    free(fzz_ext);fzz_ext = NULL;
    
    free(fxxxx);fxxxx = NULL;
    free(fyyyy);fyyyy = NULL;
    free(fzzzz);fzzzz = NULL;
    
    free(fxxyy);fxxyy = NULL;
    free(fxxzz);fxxzz = NULL;
    free(fyyzz);fyyzz = NULL;
    return;
}

void second_test3D(int Nx, int Ny, int Nz, double hx, double hy, double hz,double x0, double y0, double z0, double z1,double complex *k_bg_ext,double complex *k2_bg_ext, double complex *rhs,double complex *UE, int order,double A,double B,double C,double beta,double gamma, double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm)
{
    int i,j,l;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////// USER INPUT required : User may change the exact solution as they desired
    /////////// Input needed: Exact solution, UE_ext and user should change the rhs_ext accordingly.
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    double x,y,z,bxy_2D;
    double *fix, *fiy, *fiz;
    double complex *UE_ext, *rhs_ext;
    fix = malloc((Nx+2) * sizeof(double));
    fiy = malloc((Ny+2) * sizeof(double));
    fiz = malloc((Nz+2) * sizeof(double));
    UE_ext = malloc((Nx+2)*(Ny+2)*(Nz+2) * sizeof(double complex));
    rhs_ext = malloc((Nx+2)*(Ny+2)*(Nz+2) * sizeof(double complex));
    
    for( l = 0; l < Nx+2; l++){
        x = x0 + l*hx;
        fix[l] = sin(gamma*x);
    }
    
    for( l = 0; l < Ny+2; l++){
        y = y0 + l*hy;
        fiy[l] = sin(beta*y);
    }
    
    for( l = 0; l < Nz+2; l++){
        fiz[l] = exp(-1.0*(k_bg_ext[l])/C);
    }
    
    for( i = 0; i < Nx+2; i++) {
        for( j = 0; j < Ny+2; j++){
            bxy_2D = fix[i]*fiy[j];
            for( l = 0; l < Nz+2; l++){
                z = z0 + l*hz;
                UE_ext[i + j*(Nx+2) + l*(Nx+2)*(Ny+2)]  = fiz[l]*bxy_2D;
                rhs_ext[i + j*(Nx+2) + l*(Nx+2)*(Ny+2)] = -1.0*hz*hz*B*(2.0*A+C)*sin(C*z)*UE_ext[i + j*(Nx+2) + l*(Nx+2)*(Ny+2)];
            }
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////// END OF USER INPUT required 
    /////////// No further changes should be made beyond this point
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for( i = 0; i < Nx; i++) {
        for( j = 0; j < Ny; j++){
            for( l = 0; l < Nz; l++){
                UE[i + j*Nx + l*Nx*Ny]  = UE_ext[i+1 + (j+1)*(Nx+2) + (l+1)*(Nx+2)*(Ny+2)];
            }
        }
    }
    
    
    
    switch (order) {
        case 2:
            second_test3D_2nd(Nx,Ny,Nz,rhs,rhs_ext,UE_ext);
            break;
        case 4:
            second_test3D_4th(Nx,Ny,Nz,rhs,rhs_ext,UE_ext, ap, bp, cp, dp, am, bm, cm, dm);
            break;
        case 6:
            second_test3D_6th(Nx,Ny,Nz,hx,hy,hz,x0,y0,z0,z1,k_bg_ext,k2_bg_ext,rhs,rhs_ext,UE_ext,A,B,C,beta,gamma, a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm);
            break;
        default:
            break;
    }

    free(fix);
    fix = NULL;
    free(fiy);
    fiy = NULL;
    free(fiz);
    fiz = NULL;
    free(UE_ext);
    UE_ext = NULL;
    free(rhs_ext);
    rhs_ext = NULL;
}









