#include "../headers/structs.h"

void DST(DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out) {
 
    int i;

    for (i=0; i<dst.N; i++) { in[i] = 0.0; }

    for (i=0; i<dst.Nx; i++) { in[i+1] = creal(b[i]); }

    fftw_execute(plan);
    
    for (i=0; i<dst.Nx; i++) { bhat[i] = -cimag(out[i+1]); }
    
    for (i=0; i<dst.Nx; i++) { in[i+1] = cimag(b[i]); }

    fftw_execute(plan);

    for (i=0; i<dst.Nx; i++) { bhat[i] = dst.coef * (bhat[i] - I * cimag(out[i+1])); }
    
}
