#include "../headers/structs.h"

void residual(System sys) {

    int i,j;
    double _Complex top, middle, bottom;
    int Nx = sys.lat.Nx,  Ny = sys.lat.Ny;
    double _Complex *sol_e = malloc((Nx+2)*(Ny+2) * sizeof(double _Complex));

    for (i=0; i<(Nx+2)*(Ny+2); i++) { sol_e[i] = 0; }

    for (i=0; i<Nx; i++) {
        for (j=0; j< Ny;j++) {
            sol_e[ind_e(i,j,Nx)] = sys.sol[ind(i,j,Nx)];
        }
    }

    double _Complex ap, bp, a, b, am, bm;
    for (j=0; j<Ny; j++) {
        ap = sys.ap[j]; bp = sys.bp[j];
        a = sys.a[j]; b = sys.b[j];
        am = sys.am[j]; bm = sys.bm[j];
        for( i = 0; i < Nx; i++){
            top    = bp * sol_e[ind_e(i-1,j+1,Nx)] + ap * sol_e[ind_e(i,j+1,Nx)] + bp * sol_e[ind_e(i+1,j+1,Nx)];
            middle = b  * sol_e[ind_e(i-1,j  ,Nx)] + a  * sol_e[ind_e(i,j  ,Nx)] + b  * sol_e[ind_e(i+1,j  ,Nx)];
            bottom = bm * sol_e[ind_e(i-1,j-1,Nx)] + am * sol_e[ind_e(i,j-1,Nx)] + bm * sol_e[ind_e(i+1,j-1,Nx)];
            sys.res[ind(i,j,Nx)] = top + middle + bottom - sys.rhs[ind(i,j,Nx)];
        }
    }

    for (i=0; i<Nx*Ny; i++) { sys.error[i] = sys.sol[i] - sys.sol_analytic[i]; }

    free(sol_e); sol_e = NULL;
}
