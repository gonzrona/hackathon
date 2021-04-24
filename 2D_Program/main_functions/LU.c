#include "../headers/structs.h"

void LU(System sys) {

    int i,j,my;
    int Nx = sys.lat.Nx,  Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double _Complex *eigenValue_x = malloc(Nx * sizeof(double _Complex));
    double _Complex *eigenValue_xy = malloc(Nxy * sizeof(double _Complex));
    double _Complex *eigenValue_xyp = malloc(Nxy * sizeof(double _Complex));
    double _Complex *eigenValue_xym = malloc(Nxy * sizeof(double _Complex));

    for( i = 0; i < Nx; i++) {
        eigenValue_x[i] = 2.0 * cos((i+1)*M_PI/(Nx+1));
    }
    
#pragma omp parallel for private(i)
    for(j=0; j<Ny; j++){
        for(i=0; i<Nx; i++) {
            eigenValue_xy[ind(i,j,Nx)]  = sys.b[j]  * eigenValue_x[i] + sys.a [j];
            eigenValue_xyp[ind(i,j,Nx)] = sys.bp[j] * eigenValue_x[i] + sys.ap[j];
            eigenValue_xym[ind(i,j,Nx)] = sys.bm[j] * eigenValue_x[i] + sys.am[j];
        }
    }

    for(i=0; i<Nx; i++){
        my = i*Ny;
        sys.L[my] = 0.0;
        sys.U[my] = eigenValue_xy[i];
        sys.Up[my] = eigenValue_xyp[i];
    }

#pragma omp parallel for private(j,my)
    for(i=0; i<Nx; i++) {
        my = i*Ny;
        for( j = 1; j < Ny; j++){
            sys.L[j + my] = eigenValue_xym[ind(i,j,Nx)] / sys.U[j + my - 1] ;
            sys.U[j + my] = eigenValue_xy[ind(i,j,Nx)] - sys.L[j + my]*sys.Up[j + my - 1];
            sys.Up[j + my] = eigenValue_xyp[ind(i,j,Nx)];
        }
    }

    free(eigenValue_x); eigenValue_x = NULL;
    free(eigenValue_xy); eigenValue_xy = NULL;
    free(eigenValue_xyp); eigenValue_xyp = NULL;
    free(eigenValue_xym); eigenValue_xym = NULL;
}
