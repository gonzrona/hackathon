#include "../headers/structs.h"

#include <cufftw.h>
#include "cuda_helper.h"

void forwardDST(System sys, DSTN dst, double _Complex *rhs, double _Complex *rhat, fftw_plan plan, double *in, fftw_complex *out, fftw_plan plan2, double *in2, fftw_complex *out2) {
 
    int i,j,my;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;

#if USE_BATCHED

    int N = 2*Nx + 2, NC = (N/2) + 1;
#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;

        for (i=0; i<dst.Nx; i++) { in[(j*N) + i+1] = creal(rhs[i + my]); }
        for (i=0; i<dst.Nx; i++) { in2[(j*N) + i+1] = cimag(rhs[i + my]); }       
    }

    fftw_execute(plan); /********************* FFTW *********************/
    fftw_execute(plan2); /********************* FFTW *********************/

#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;

        for (i=0; i<dst.Nx; i++) { rhat[i + my] = dst.coef * (-cimag(out[(j*NC) + i+1]) - I * cimag(out2[(j*NC) + i+1])); }
    }

#else

#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;

        for (i=0; i<dst.Nx; i++) { in[i+1] = creal(rhs[i + my]); }
        for (i=0; i<dst.Nx; i++) { in2[i+1] = cimag(rhs[i + my]); }
        
        fftw_execute(plan); /********************* FFTW *********************/
        fftw_execute(plan2); /********************* FFTW *********************/

        for (i=0; i<dst.Nx; i++) { rhat[i + my] = dst.coef * (-cimag(out[i+1]) - I * cimag(out2[i+1])); }
    }

#endif
}

void reverseDST(System sys, DSTN dst, double _Complex *xhat, double _Complex *sol, fftw_plan plan, double *in, fftw_complex *out, fftw_plan plan2, double *in2, fftw_complex *out2) {
 
    int i,j,my;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;

#if USE_BATCHED    

    int N = 2*Nx + 2, NC = (N/2) + 1;
#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;

        for (i=0; i<dst.Nx; i++) { in[(j*N) + i+1] = creal(xhat[j + i*Ny]); }
        for (i=0; i<dst.Nx; i++) { in2[(j*N) + i+1] = cimag(xhat[j + i*Ny]); }
    }

        fftw_execute(plan); /********************* FFTW *********************/
        fftw_execute(plan2); /********************* FFTW *********************/

#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;

        for (i=0; i<dst.Nx; i++) { sol[i + my] = dst.coef * (-cimag(out[(j*NC) + i+1]) - I * cimag(out2[(j*NC) + i+1])); }
        
    }

#else

#pragma omp for
    for(j = 0; j < Ny; j++) {
        my = j*Nx;

        for (i=0; i<dst.Nx; i++) { in[i+1] = creal(xhat[j + i*Ny]); }
        for (i=0; i<dst.Nx; i++) { in2[i+1] = cimag(xhat[j + i*Ny]); }
        
        fftw_execute(plan); /********************* FFTW *********************/
        fftw_execute(plan2); /********************* FFTW *********************/

        for (i=0; i<dst.Nx; i++) { sol[i + my] = dst.coef * (-cimag(out[i+1]) - I * cimag(out2[i+1])); }
    }

#endif
}
