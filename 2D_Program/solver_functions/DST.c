#include "../headers/structs.h"

void forwardDST(System sys, DSTN dst, double _Complex *rhs, double _Complex *rhat, fftw_plan plan, double *in, fftw_complex *out) {
 
    int i,j,my;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    
#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;
        
        for (i=0; i<dst.N; i++) { in[i] = 0.0; }

        for (i=0; i<dst.Nx; i++) { in[i+1] = creal(rhs[i + my]); }
        
        fftw_execute(plan); /********************* FFTW *********************/
        
        for (i=0; i<dst.Nx; i++) { rhat[i + my] = -cimag(out[i+1]); }

        for (i=0; i<dst.Nx; i++) { in[i+1] = cimag(rhs[i + my]); }

        fftw_execute(plan); /********************* FFTW *********************/

        for (i=0; i<dst.Nx; i++) { rhat[i + my] = dst.coef * (rhat[i + my] - I * cimag(out[i+1])); }
        
    }
}

void reverseDST(System sys, DSTN dst, double _Complex *xhat, double _Complex *sol, fftw_plan plan, double *in, fftw_complex *out) {
 
    int i,j,my;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    
#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;
        
        for (i=0; i<dst.N; i++) { in[i] = 0.0; }

        for (i=0; i<dst.Nx; i++) { in[i+1] = creal(xhat[j + i*Ny]); }
        
        fftw_execute(plan); /********************* FFTW *********************/
        
        for (i=0; i<dst.Nx; i++) { sol[i + my] = -cimag(out[i+1]); }

        for (i=0; i<dst.Nx; i++) { in[i+1] = cimag(xhat[j + i*Ny]); }

        fftw_execute(plan); /********************* FFTW *********************/

        for (i=0; i<dst.Nx; i++) { sol[i + my] = dst.coef * (sol[i + my] - I * cimag(out[i+1])); }
        
    }
}
