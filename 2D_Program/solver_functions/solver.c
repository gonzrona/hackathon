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


void DST(DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out);

void solver(System sys) {

    PUSH_RANGE("solver", 0)
    
    DSTN dst;
    int i,j,my,mx;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double _Complex *rhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    double _Complex *xhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    
    int N = 2*Nx + 2, NC = (N/2) + 1;
    dst.Nx = Nx; dst.N = N; dst.coef = sqrt(2.0/(Nx+1));

    printf("Nx = %d: Ny = %d: Nxy = %d: N = %d: NC = %d\n", Nx, Ny, Nxy, N, NC);
    
#if USE_OMP
#pragma omp parallel private (i,j,mx,my)
    {
#endif
            
        double *in        = (double *) fftw_malloc(sizeof(double) * N); /********************* FFTW *********************/
        printf("Size of in = %lu\n", sizeof(double) * N);
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NC); /********************* FFTW *********************/

        double _Complex *b    = (double _Complex *) malloc(Nx * sizeof(double _Complex));
        double _Complex *bhat = (double _Complex *) malloc(Nx * sizeof(double _Complex));
        double _Complex *y    = (double _Complex *) malloc(Ny * sizeof(double _Complex));
        fftw_plan plan; /********************* FFTW *********************/

    PUSH_RANGE("1st fffw_plan", 1)     
#if USE_OMP   
    #pragma omp critical (make_plan)
#endif
        { plan = fftw_plan_dft_r2c_1d ( N, in, out, FFTW_ESTIMATE ); } /********************* FFTW *********************/
        // { plan2 = fftw_plan_many_dft_r2c ( 1, n, Ny, in, inemded, 1, Nx, out, outemded, 1, Nx, FFTW_MEASURE ); } /********************* FFTW *********************/
    POP_RANGE

    PUSH_RANGE("1st DST", 1)
#if USE_OMP
    #pragma omp for  
#endif  
        for(j = 0; j < Ny; j++) {
            my = j*Nx;
            for(i = 0; i < Nx; i++){
                b[i] = sys.rhs[i + my];
            }
            DST(dst, b, bhat, plan, in, out); /********************* FFTW contained inside *********************/
            for(i = 0; i < Nx; i++){
                rhat[i + my] = bhat[i];
            }
        }
    POP_RANGE
        
    PUSH_RANGE("Middle stuff", 2)
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
      
    PUSH_RANGE("2nd DST", 3)
#if USE_OMP
    #pragma omp for
#endif
        for(j = 0; j < Ny; j++) {
            my = j*Nx;
            for(i = 0; i < Nx; i++){
                b[i] = xhat[j + i*Ny];
            }
            DST(dst, b, bhat, plan, in, out); /********************* FFTW contained inside *********************/
            for(i = 0; i < Nx; i++){
                sys.sol[i + my] = bhat[i];
            }
        }
    POP_RANGE
        
        PUSH_RANGE("Cleanup", 4)
        fftw_destroy_plan(plan); /********************* FFTW *********************/
        free(in); in = NULL;
        fftw_free(out); out = NULL; /********************* FFTW *********************/
        free(b); b = NULL;
        free(bhat); bhat = NULL;
        free(y); y = NULL;
#if USE_OMP
    }
#endif

    free(rhat); rhat = NULL;
    free(xhat); xhat = NULL;
    POP_RANGE

    POP_RANGE
}
