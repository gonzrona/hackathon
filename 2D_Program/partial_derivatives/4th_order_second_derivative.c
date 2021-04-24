#include "../headers/structs.h"
#include "../headers/prototypes.h"

void f_xx_4th_order(System sys, double _Complex *f_e, double _Complex *fxx_e) {
    int i, j, Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double x, y, hx = sys.lat.hx, hy = sys.lat.hy, x0 = sys.lat.x0, x1 = sys.lat.x1, y0 = sys.lat.y0, y1 = sys.lat.y1;
    double _Complex *L   = malloc(Nxy  * sizeof(double _Complex));
    double _Complex *U   = malloc(Nxy  * sizeof(double _Complex));
    double _Complex *fxx = malloc(Nxy  * sizeof(double _Complex));

#pragma omp parallel for private(i,y)
    for (j=0; j<Ny; j++) {
        for (i=0; i<Nx; i++) {
            fxx[ind(i,j,Nx)] = 12.0*(f_e[ind_e(i+1,j,Nx)] - 2.0*f_e[ind_e(i,j,Nx)] + f_e[ind_e(i-1,j,Nx)])/hx/hx;
        }

        y = y0 + (j+1)*hy;
        fxx[ind(0,j,Nx)] = fxx[ind(0,j,Nx)] - f_xx(x0,y);
        fxx[ind(Nx-1,j,Nx)] = fxx[ind(Nx-1,j,Nx)] - f_xx(x1,y);

        U[ind(0,j,Nx)] = 10.0;

        for(i=1; i<Nx; i++){
            L[ind(i-1,j,Nx)] = 1.0/U[ind(i-1,j,Nx)];
            U[ind(i,j,Nx)] = 10.0 - L[ind(i-1,j,Nx)];
        }

        fxx[ind(0,j,Nx)] = fxx[ind(0,j,Nx)];

        for(i=1; i<Nx; i++) {
            fxx[ind(i,j,Nx)] = fxx[ind(i,j,Nx)] - L[ind(i-1,j,Nx)]*fxx[ind(i-1,j,Nx)];
        }

        fxx[ind(Nx-1,j,Nx)] = fxx[ind(Nx-1,j,Nx)]/U[ind(Ny-1,j,Nx)];

        for (i=Nx-2; i>=0; i--) {
            fxx[ind(i,j,Nx)] = (fxx[ind(i,j,Nx)]-fxx[ind(i+1,j,Nx)])/U[ind(i,j,Nx)];
        }
    }
    
    for (i=0; i<Nx+2; i++) {
        x = x0 + i*hx;
        fxx_e[ind(i,0,Nx+2)] = f_xx(x,y0);
        fxx_e[ind(i,Ny+1,Nx+2)] = f_xx(x,y1);
    }
    
    for (j=0; j<Ny+2; j++) {
        y = y0 + j*hy;
        fxx_e[ind(0,j,Nx+2)] = f_xx(x0,y);
        fxx_e[ind(Nx+1,j,Nx+2)] = f_xx(x1,y);
    }
    
    for (i=0; i<Nx; i++) {
        for (j=0; j<Ny; j++) {
            fxx_e[ind_e(i,j,Nx)] = fxx[ind(i,j,Nx)];
        }
    }

    free(L); L = NULL;
    free(U); U = NULL;
    free(fxx); fxx = NULL;
}

void f_yy_4th_order(System sys, double _Complex *f_e, double _Complex *fyy_e) {
    int i, j, Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double x, y, hx = sys.lat.hx, hy = sys.lat.hy, x0 = sys.lat.x0, x1 = sys.lat.x1, y0 = sys.lat.y0, y1 = sys.lat.y1;
    double _Complex *L   = malloc(Nxy  * sizeof(double _Complex));
    double _Complex *U   = malloc(Nxy  * sizeof(double _Complex));
    double _Complex *fyy = malloc(Nxy  * sizeof(double _Complex));

#pragma omp parallel for private(j,x)
    for (i=0; i<Nx; i++) {
        for (j=0; j<Ny; j++) {
            fyy[ind(i,j,Nx)] = 12.0*(f_e[ind_e(i,j+1,Nx)] - 2.0*f_e[ind_e(i,j,Nx)] + f_e[ind_e(i,j-1,Nx)])/hy/hy;
        }

        x = x0 + (i+1)*hx;
        fyy[ind(i,0,Nx)] = fyy[ind(i,0,Nx)] - f_yy(x,y0);
        fyy[ind(i,Ny-1,Nx)] = fyy[ind(i,Ny-1,Nx)] - f_yy(x,y1);

        U[ind(i,0,Nx)] = 10.0;

        for(j=1; j<Ny; j++){
            L[ind(i,j-1,Nx)] = 1.0/U[ind(i,j-1,Nx)];
            U[ind(i,j,Nx)] = 10.0 - L[ind(i,j-1,Nx)];
        }

        fyy[ind(i,0,Nx)] = fyy[ind(i,0,Nx)];
        
        for(j=1; j<Ny; j++) {
            fyy[ind(i,j,Nx)] = fyy[ind(i,j,Nx)] - L[ind(i,j-1,Nx)]*fyy[ind(i,j-1,Nx)];
        }
 
        fyy[ind(i,Ny-1,Nx)] = fyy[ind(i,Ny-1,Nx)]/U[ind(i,Ny-1,Nx)];

        for (j=Ny-2; j>=0; j--) {
            fyy[ind(i,j,Nx)] = (fyy[ind(i,j,Nx)]-fyy[ind(i,j+1,Nx)])/U[ind(i,j,Nx)];
        }
    }

    for (i=0; i<Nx+2; i++) {
        x = x0 + i*hx;
        fyy_e[ind(i,0,Nx+2)] = f_yy(x,y0);
        fyy_e[ind(i,Ny+1,Nx+2)] = f_yy(x,y1);
    }
    
    for (j=0; j<Ny+2; j++) {
        y = y0 + j*hy;
        fyy_e[ind(0,j,Nx+2)] = f_yy(x0,y);
        fyy_e[ind(Nx+1,j,Nx+2)] = f_yy(x1,y);
    }
    
    for (i=0; i<Nx; i++) {
        for (j=0; j<Ny; j++) {
            fyy_e[ind_e(i,j,Nx)] = fyy[ind(i,j,Nx)];
        }
    }

    free(L); L = NULL;
    free(U); U = NULL;
    free(fyy); fyy = NULL;
}
