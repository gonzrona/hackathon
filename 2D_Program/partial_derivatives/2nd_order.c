#include "../headers/structs.h"
#include "../headers/prototypes.h"

void derivative_xx_2nd_order(System sys, double _Complex *f, double _Complex *fxx) {
    int i, j, Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    double hx2 = sys.lat.hx*sys.lat.hx;
#pragma omp parallel for private(i)
    for (j=0; j<Ny; j++) {
        for (i=0; i<Nx; i++) {
            fxx[ind(i,j,Nx)] = (f[ind_e(i-1,j,Nx)] - 2.0*f[ind_e(i,j,Nx)] + f[ind_e(i+1,j,Nx)])/hx2;
        }
    }
}

void derivative_yy_2nd_order(System sys, double _Complex *f, double _Complex *fyy) {
    int i, j, Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    double hy2 = sys.lat.hy*sys.lat.hy;
#pragma omp parallel for private(i)
    for (j=0; j<Ny; j++) {
        for (i=0; i<Nx; i++) {
            fyy[ind(i,j,Nx)] = (f[ind_e(i,j-1,Nx)] - 2.0*f[ind_e(i,j,Nx)] + f[ind_e(i,j+1,Nx)])/hy2;
        }
    }
}
