#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>
#include <sys/time.h>

void coefficients(double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, double complex *k2_bg, double hx, double hy, double hz, int Nz, int order, double A,double B,double C, double z0, double z1);

void background3D(int Nz, double hz, double z0, double complex *k_bg_ext, double complex *k2_bg_ext, double A,double B,double C);

void second_test3D(int Nx, int Ny, int Nz, double hx, double hy, double hz, double x0, double y0, double z0,double z1, double complex *k_bg_ext, double complex *k2_bg_ext, double complex *rhs,double complex *UE, int order, double A,double B,double C,double beta,double gamma, double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm);

void LU(double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, int Nx, int Ny, int Nz, double hx, double hy, double hz, double complex *k2_bg_ext, double complex *DL,double complex *DU, double complex *DC);

void solver_3D_dir_Z_DDD(int Nx, int Ny, int Nz, double complex *SOL, double complex *DL, double complex *DU,double complex *DM, double complex *rhs);

void residual3D(int Nx, int Ny, int Nz, double hx, double hy, double hz, double complex *SOL, double complex *res,double complex *rhs, double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm,double complex *k2_bg_ext );

double normL2(int N, double complex *x);
double L2err(int N, double complex *x, double complex *u);

double cpuSecond(void) ;

int main(int argc, char **argv){
    
    int Nx, Ny, Nz, Nxyz;
    double x0,y0,z0,x1,y1,z1,hx,hy,hz,norm;
    double complex *k_bg_ext, *k2_bg_ext;
    double complex *UE, *rhs;
    double complex *DL, *DU, *DC;
    double complex *SOL, *res;
    double complex *a, *b, *c, *d;
    double complex *ap, *bp, *cp, *dp;
    double complex *am, *bm, *cm, *dm;


    /*
     Computational Domain
     */
    
    x0 = 0.0 ; y0 = 0.0 ; z0 = 0.0;
    x1 = M_PI; y1 = M_PI; z1 = M_PI;
    
    /*
     Nx, Ny, Nz - the number of grid points in x-, y- and z- directions,
     Nxyz = Nx*Ny*Nz - the size of the unknown vector, hx, hy,hz - the grid
     sizes in x-, y- and z- directions.
     */
    
    Nx=4;
    Ny=4;
    Nz=9;
    
    int order = 2;
    if (argc ==2) {
        order = atoi(argv[1]);
    }
    else if (argc == 3){
        order = atoi(argv[1]);
        Nx = atoi(argv[2]);
        Ny = Nx;
        Nz = Nx;
    }
    
    Nxyz=Nx*Ny*Nz;
    hx=(x1-x0)/(Nx+1);
    hy=(y1-y0)/(Ny+1);
    hz=(z1-z0)/(Nz+1);
    
    /*
     Make sure the program is using 2nd, 4th or 6th order. If 6th is requested without uniform grid, drop to 4th order.
     */
    if (order==6) {
        if (Nx!=Ny||Ny!=Nz||Nx!=Nz) {
            printf("\n\tNOTE:\tThe use of 6th order requires a uniform grid. So the order\n\t\tof approximation has been reduced to 4th to accommodate this grid.\n\n");
            order = 4;
        }
    }
    else if(order !=2 && order!=4){
        printf("\n\tNOTE:\tThis program requires the use of 2nd, 4th or 6th order approximation.\n\t\tThus the default, 2nd order, is being used.\n\n");
        order = 2;
    }
    
    printf("\n\tGrid: %d x %d x %d with order: %d\n\n",Nx,Ny,Nz,order);
    
    /*
     Allocate and define the background coefficient
     */

    /*
    Parameter used of the test problem with k(z) = A-B*sin(C*z)

    Note:
    For this code it is necessary that both beta and gamma have to be integer to satisfy the boundary problem
    */

    double A,B,C,beta,gamma;
    A = 10.0;
    B = 0.0;//9.0;
    C = 10.0;
    gamma = 6.0;//9.0;
    beta = sqrt(A*A+B*B-gamma*gamma);
    k_bg_ext = malloc((Nz+2) * sizeof(double complex));
    k2_bg_ext = malloc((Nz+2) * sizeof(double complex));
    background3D(Nz, hz, z0, k_bg_ext, k2_bg_ext,A,B,C);

    /*
     Allocate and define the stencil coefficients
     */
    a = malloc(Nz * sizeof(double complex));
    b = malloc(Nz * sizeof(double complex));
    c = malloc(Nz * sizeof(double complex));
    d = malloc(Nz * sizeof(double complex));
    ap = malloc(Nz * sizeof(double complex));
    bp = malloc(Nz * sizeof(double complex));
    cp = malloc(Nz * sizeof(double complex));
    dp = malloc(Nz * sizeof(double complex));
    am = malloc(Nz * sizeof(double complex));
    bm = malloc(Nz * sizeof(double complex));
    cm = malloc(Nz * sizeof(double complex));
    dm = malloc(Nz * sizeof(double complex));
    coefficients(a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm, k2_bg_ext, hx, hy, hz, Nz, order,A,B,C,z0,z1);
    
    /*
     Allocate and define analytic solution UE and the corresponding right hand side r.h.s
     for the test problem.
     */
    UE  = malloc(Nxyz * sizeof(double complex));
    rhs = malloc(Nxyz * sizeof(double complex));
    second_test3D(Nx,Ny,Nz,hx,hy,hz,x0,y0,z0,z1, k_bg_ext, k2_bg_ext,rhs,UE,order,A,B,C,beta,gamma,a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm);
    
    /*
     Find the LU factorization of the transformed system: DL and DU and
     the main diagonal DM_S in the original (untransformed) system.
     */
    DL    = malloc(Nxyz  * sizeof(double complex));
    DU    = malloc(Nxyz  * sizeof(double complex));
    DC    = malloc(Nxyz  * sizeof(double complex));
    
    LU(a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm, Nx, Ny, Nz, hx, hy, hz, k2_bg_ext,DL,DU,DC);
    
    /*
     Direct solution by using Fourier transform in x- and y- directions and
     the direct tridiagonal solver in y-direction.
     */
    
    SOL  = malloc(Nxyz * sizeof(double complex));
    solver_3D_dir_Z_DDD(Nx, Ny, Nz, SOL, DL, DU, DC, rhs);

    /// Calculating the inf-norm of the difference between exact and calculated solution.
    int i;
    double max = 0;
    double diff = 0;
    for (i=0; i<Nxyz; i++) {
        diff = cabs(UE[i]-SOL[i]);
        if (diff>max) {
            max=diff;
        }
    }
    norm = L2err(Nxyz, UE, SOL);
    printf("\t||error||_inf = %10.7e \n\n",max);
    printf("\t||error||_L2_err =  %10.7e \n\n",norm);
    
    /*
     Calculate the residual A*SOL-rhs for finding redidual
     */
    
    res  = malloc(Nxyz * sizeof(double complex));
    residual3D(Nx,Ny,Nz,hx,hy,hz,SOL,res,rhs,a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm,k2_bg_ext);
    
    
    /*
     Calculate the l2-norm of the residual
     */
    norm = normL2(Nxyz, res);
    printf("\t||res||_2 =  %10.7e \n\n",norm);
    
    free(k_bg_ext);
    k_bg_ext = NULL;
    free(k2_bg_ext);
    k2_bg_ext = NULL;
    free(UE);
    UE = NULL;
    free(rhs);
    rhs = NULL;
    
    free(DL);
    DL = NULL;
    free(DU);
    DU = NULL;
    free(SOL);
    SOL = NULL;
    free(res);
    res = NULL;
    
    
    free(a);free(ap);free(am);
    a = NULL; ap = NULL; am = NULL;
    free(b);free(bp);free(bm);
    b = NULL; bp = NULL; bm = NULL;
    free(c);free(cp);free(cm);
    c = NULL; cp = NULL; cm = NULL;
    free(d);free(dp);free(dm);
    d = NULL; dp = NULL; dm = NULL;

    free(DC);
    DC = NULL;
    
    return 0;
} /* end function main */

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ( (double)tp.tv_sec + (double) tp.tv_usec*1.e-6 );
}



