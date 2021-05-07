#include "../headers/structs.h"

#include <cufftw.h>
#include <nvToolsExt.h>

//*****************************************************
//                        NVTX                         
//*****************************************************
#include<stdint.h>
static const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
//*****************************************************
//                        NVTX                         
//*****************************************************

// void DST(DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out) {
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
