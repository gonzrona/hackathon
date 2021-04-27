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

void DST(DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out) {
 
    int i;

    PUSH_RANGE("reset in", 5)
    for (i=0; i<dst.N; i++) { in[i] = 0.0; }
    POP_RANGE

    PUSH_RANGE("creal(b[i])", 6)
    for (i=0; i<dst.Nx; i++) { in[i+1] = creal(b[i]); }
    POP_RANGE

    PUSH_RANGE("1st fffw_execute", 7)
    fftw_execute(plan); /********************* FFTW *********************/
    POP_RANGE
    
    PUSH_RANGE("-cimag(out[i+1])", 8)
    for (i=0; i<dst.Nx; i++) { bhat[i] = -cimag(out[i+1]); }
    POP_RANGE
    
    PUSH_RANGE("cimag(b[i])", 9)
    for (i=0; i<dst.Nx; i++) { in[i+1] = cimag(b[i]); }
    POP_RANGE

    PUSH_RANGE("2nd fffw_execute", 10)
    fftw_execute(plan); /********************* FFTW *********************/
    POP_RANGE

    PUSH_RANGE("bhat[i]", 11)
    for (i=0; i<dst.Nx; i++) { bhat[i] = dst.coef * (bhat[i] - I * cimag(out[i+1])); }
    POP_RANGE
    
}
