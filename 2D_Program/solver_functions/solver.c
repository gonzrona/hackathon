#include "../headers/structs.h"

#include <assert.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
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

//*******************************************************
// *************** FOR ERROR CHECKING *******************
//*******************************************************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        cudaError_t status = (cudaError_t)( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
//*******************************************************
// *************** FOR ERROR CHECKING *******************
//*******************************************************

void DST(DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out);
void forwardDST(System sys, DSTN dst, double _Complex *rhs, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out, fftw_plan plan2, double *in2, fftw_complex *out2);
void reverseDST(System sys, DSTN dst, double _Complex *xhat, double _Complex *sol, fftw_plan plan, double *in, fftw_complex *out, fftw_plan plan2, double *in2, fftw_complex *out2);


void solver(System sys) {

    PUSH_RANGE("solver", 0)
    
    DSTN dst;
    int i,j,mx;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double _Complex *rhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    double _Complex *xhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));

    int N = 2*Nx + 2, NC = (N/2) + 1;
    dst.Nx = Nx; dst.N = N; dst.coef = sqrt(2.0/(Nx+1));

    printf("Nx = %d: Ny = %d: Nxy = %d: N = %d: NC = %d\n", Nx, Ny, Nxy, N, NC);
    
#if USE_OMP
#pragma omp parallel private (i,j,mx)
    {
#endif
            
#if USE_CUFFTW
        double *in, *in2;
        fftw_complex *out, *out2;
        CUDA_RT_CALL(cudaMallocHost((void**)&in, sizeof(double) * N));
        CUDA_RT_CALL(cudaMallocHost((void**)&in2, sizeof(double) * N));
        CUDA_RT_CALL(cudaMallocHost((void**)&out, sizeof(fftw_complex) * NC));
        CUDA_RT_CALL(cudaMallocHost((void**)&out2, sizeof(fftw_complex) * NC));
#else
        double *in        = (double *) fftw_malloc(sizeof(double) * N); /********************* FFTW *********************/
        double *in2        = (double *) fftw_malloc(sizeof(double) * N); /********************* FFTW *********************/
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NC); /********************* FFTW *********************/
        fftw_complex *out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NC); /********************* FFTW *********************/
#endif

    memset(in, 0, sizeof(double) * N);
    memset(in2, 0, sizeof(double) * N);
    memset(out, 0, sizeof(fftw_complex) * NC);
    memset(out2, 0, sizeof(fftw_complex) * NC);

    double _Complex *y    = (double _Complex *) malloc(Ny * sizeof(double _Complex));
    fftw_plan plan, plan2; /********************* FFTW *********************/

    PUSH_RANGE("1st fffw_plan", 1)     
#if USE_OMP   
    #pragma omp critical (make_plan)
#endif
    plan = fftw_plan_dft_r2c_1d ( N, in, out, FFTW_ESTIMATE ); /********************* FFTW *********************/
    plan2 = fftw_plan_dft_r2c_1d ( N, in2, out2, FFTW_ESTIMATE ); /********************* FFTW *********************/
    POP_RANGE

    PUSH_RANGE("forwardDST", 2)
    forwardDST(sys, dst, sys.rhs, rhat, plan, in, out, plan2, in2, out2);
    POP_RANGE

        
    PUSH_RANGE("Middle stuff", 3)
#if USE_OMP
    #pragma omp for
#endif
        for(i = 0; i < Nx; i++){
            y[0] = rhat[i];
            mx = i*Ny ;
            for(j = 1; j < Ny; j++) {
                y[j] = rhat[ind(i,j,Nx)] - sys.L[j + mx]*y[j - 1];
            }
            xhat[Ny - 1 + mx] = y[Ny - 1]/sys.U[Ny - 1 + mx] ;
            for(j = Ny-2; j >= 0; j--) {
                xhat[j + mx] =  ( y[j] - sys.Up[j + mx] * xhat[j + 1 + mx] )/sys.U[j + mx] ;
            }
        }
    POP_RANGE
      
    PUSH_RANGE("reverseDST", 4)
    reverseDST(sys, dst, xhat, sys.sol, plan, in, out, plan2, in2, out2);
    POP_RANGE

    PUSH_RANGE("Cleanup", 5)
#if USE_CUFFTW
    CUDA_RT_CALL(cudaFreeHost(in));
    CUDA_RT_CALL(cudaFreeHost(out));
    CUDA_RT_CALL(cudaFreeHost(in2));
    CUDA_RT_CALL(cudaFreeHost(out2));
#else
    free(in); in = NULL;
    fftw_free(out); out = NULL; /********************* FFTW *********************/
    free(in2); in = NULL;
    fftw_free(out2); out2 = NULL; /********************* FFTW *********************/
#endif

    fftw_destroy_plan(plan); /********************* FFTW *********************/
    free(y); y = NULL;

#if USE_OMP
    }
#endif    

    free(rhat); rhat = NULL;
    free(xhat); xhat = NULL;
    POP_RANGE

    POP_RANGE
}
