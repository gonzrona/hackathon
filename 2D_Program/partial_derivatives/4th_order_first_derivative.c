#include "../headers/structs.h"
#include "../headers/prototypes.h"

void k2_y_4th_order(System sys, double _Complex *k2_e, double _Complex *k2y) {
    int j, Ny = sys.lat.Ny;
    double hy = sys.lat.hy, y0 = sys.lat.y0, y1 = sys.lat.y1;
    double _Complex *L = malloc(Ny  * sizeof(double _Complex));
    double _Complex *U = malloc(Ny  * sizeof(double _Complex));

    for (j=0; j<Ny; j++) {
        k2y[j] = 3.0*(k2_e[j+2]-k2_e[j])/hy;
    }
    
    k2y[0] = k2y[0] - k2_y(y0);
    k2y[Ny-1] = k2y[Ny-1] - k2_y(y1);
    
    U[0] = 4.0;
    for(j=1; j<Ny; j++) {
        L[j-1] = 1.0/U[j-1];
        U[j] = 4.0 - L[j-1];
    }
    
    for(j=1; j<Ny; j++) {
        k2y[j] = k2y[j] - L[j-1]*k2y[j-1];
    }
    
    k2y[Ny-1] = k2y[Ny-1]/U[Ny-1];
    for (j=Ny-2; j>=0; j--) {
        k2y[j] = (k2y[j]-k2y[j+1])/U[j];
    }

    free(L); L = NULL;
    free(U); U = NULL;
}

void f_y_4th_order(System sys, double _Complex *f_e, double _Complex *fy) {
    int i, j, Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double x, hx = sys.lat.hx, hy = sys.lat.hy, x0 = sys.lat.x0, y0 = sys.lat.y0, y1 = sys.lat.y1;
    double _Complex *L = malloc(Nxy  * sizeof(double _Complex));
    double _Complex *U = malloc(Nxy  * sizeof(double _Complex));
    
#pragma omp parallel for private(j,x)
    for (i=0; i<Nx; i++) {
        for (j=0; j<Ny; j++) {
            fy[ind(i,j,Nx)] = 3.0*(f_e[ind_e(i,j+1,Nx)]-f_e[ind_e(i,j-1,Nx)])/hy;
        }

        x = x0 + (i+1)*hx;
        fy[ind(i,0,Nx)] = fy[ind(i,0,Nx)] - f_y(x,y0);
        fy[ind(i,Ny-1,Nx)] = fy[ind(i,Ny-1,Nx)] - f_y(x,y1);
    
        U[ind(i,0,Nx)] = 4.0;
        
        for(j=1; j<Ny; j++){
            L[ind(i,j-1,Nx)] = 1.0/U[ind(i,j-1,Nx)];
            U[ind(i,j,Nx)] = 4.0 - L[ind(i,j-1,Nx)];
        }
        
        fy[ind(i,0,Nx)] = fy[ind(i,0,Nx)];
  
        for(j=1; j<Ny; j++) {
            fy[ind(i,j,Nx)] = fy[ind(i,j,Nx)] - L[ind(i,j-1,Nx)]*fy[ind(i,j-1,Nx)];
        }
  
        fy[ind(i,Ny-1,Nx)] = fy[ind(i,Ny-1,Nx)]/U[ind(i,Ny-1,Nx)];
   
        for (j=Ny-2; j>=0; j--) {
            fy[ind(i,j,Nx)] = (fy[ind(i,j,Nx)]-fy[ind(i,j+1,Nx)])/U[ind(i,j,Nx)];
        }
    }

    free(L); L = NULL;
    free(U); U = NULL;
}
