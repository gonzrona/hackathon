#include "../headers/structs.h"

#include <assert.h>
#include <string.h>

#include <cufftw.h>
#include "cuda_helper.h"

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

#if USE_BATCHED
    size_t size_in = sizeof(double) * N * Ny;
    size_t size_out = sizeof(fftw_complex) * NC * Ny;
#else 
    size_t size_in = sizeof(double) * N;
    size_t size_out = sizeof(fftw_complex) * NC;
#endif
            
#if USE_CUFFTW
        double *in, *in2;
        fftw_complex *out, *out2;
        CUDA_RT_CALL(cudaMallocHost((void**)&in, size_in));
        CUDA_RT_CALL(cudaMallocHost((void**)&in2, size_in));
        CUDA_RT_CALL(cudaMallocHost((void**)&out, size_out));
        CUDA_RT_CALL(cudaMallocHost((void**)&out2, size_out));
#else
        double *in        = (double *) fftw_malloc(size_in); /********************* FFTW *********************/
        double *in2        = (double *) fftw_malloc(size_in); /********************* FFTW *********************/
        fftw_complex *out = (fftw_complex*) fftw_malloc(size_out); /********************* FFTW *********************/
        fftw_complex *out2 = (fftw_complex*) fftw_malloc(size_out); /********************* FFTW *********************/
#endif

    memset(in, 0, size_in);
    memset(in2, 0, size_in);
    memset(out, 0, size_out);
    memset(out2, 0, size_out);

#if USE_BATCHED
    /**********************BATCHED***************************/
    int rank = 1; /* not 2: we are computing 1d transforms */
    int n[] = {N};
    int howmany = Ny;
    int idist = N;
    int odist = NC;
    int istride = 1;
    int ostride = 1; /* distance between two elements in the same column */
    int *inembed = NULL;
    int *onembed = NULL;
    /**********************BATCHED***************************/
#endif
    
    fftw_plan plan, plan2; /********************* FFTW *********************/
    double _Complex *y    = (double _Complex *) malloc(Ny * sizeof(double _Complex));
    

    PUSH_RANGE("1st fffw_plan", 1)     
#if USE_OMP   
    #pragma omp critical (make_plan)
#endif
    {
#if USE_BATCHED
    plan = fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_ESTIMATE);
    plan2 = fftw_plan_many_dft_r2c(rank, n, howmany, in2, inembed, istride, idist, out2, onembed, ostride, odist, FFTW_ESTIMATE);
#else
    plan = fftw_plan_dft_r2c_1d ( N, in, out, FFTW_ESTIMATE ); /********************* FFTW *********************/
    plan2 = fftw_plan_dft_r2c_1d ( N, in2, out2, FFTW_ESTIMATE ); /********************* FFTW *********************/
#endif
    }
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
    fftw_destroy_plan(plan2); /********************* FFTW *********************/
    free(y); y = NULL;

#if USE_OMP
    }
#endif    

    free(rhat); rhat = NULL;
    free(xhat); xhat = NULL;
    POP_RANGE

    POP_RANGE
}
