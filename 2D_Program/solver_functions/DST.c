#include "../headers/structs.h"


void DST(enum Direction dir, System sys, DSTN dst, double _Complex *x, double _Complex *xhat) {
 
    int i,j;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny, N = dst.N;
    int NC = dst.NC;
    
    double *in1 = dst.in1, *in2 = dst.in2;
    fftw_complex *out1 = dst.out1, *out2 = dst.out2;
    fftw_plan plan1 = dst.plan1, plan2 = dst.plan2;

    for(i=0; i<N*Ny; i++) {
        in1[i] = 0.;
        in2[i] = 0.;
    }
    
    if (dir == forward) {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                in1[ind(i+1,j,N)] = creal(x[ind(i,j,Nx)]);
                in2[ind(i+1,j,N)] = cimag(x[ind(i,j,Nx)]);
            }
        }
    }
    else {
        for (j=0; j<Ny; j++) {
            for (i=0; i<Nx; i++) {
                in1[ind(i+1,j,N)] = creal(x[ind(j,i,Ny)]);
                in2[ind(i+1,j,N)] = cimag(x[ind(j,i,Ny)]);
            }
        }
    }


    fftw_execute(plan1);
    fftw_execute(plan2);
    
    
    for (i=0; i<Nx; i++) {
        for (j=0; j<Ny; j++) {
            xhat[ind(i,j,Nx)] = -dst.coef * (cimag(out1[ind(i+1,j,NC)]) + I * cimag(out2[ind(i+1,j,NC)]));
        }
    }
}
