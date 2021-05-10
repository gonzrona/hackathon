#include "../headers/structs.h"

#include <assert.h>
#include <string.h>

#include <cuComplex.h>
#include <cufftw.h>

#include "cuda_helper.h"
#include "cuda_kernels.h"

void DST( DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out );
void forwardDST( System           sys,
                 DSTN             dst,
                 cuDoubleComplex *rhs,
                 cuDoubleComplex *bhat,
                 fftw_plan        plan,
                 double *         in,
                 fftw_complex *   out,
                 fftw_plan        plan2,
                 double *         in2,
                 fftw_complex *   out2 );
void reverseDST( System           sys,
                 DSTN             dst,
                 cuDoubleComplex *xhat,
                 cuDoubleComplex *sol,
                 fftw_plan        plan,
                 double *         in,
                 fftw_complex *   out,
                 fftw_plan        plan2,
                 double *         in2,
                 fftw_complex *   out2 );
void fullDST( const System           sys,
              const DSTN             dst,
              const fftw_plan        plan,
              const fftw_plan        plan2,
              cuDoubleComplex *d_rhat,
              cuDoubleComplex *d_xhat,
              cuDoubleComplex *d_y,
              double *         in,
              fftw_complex *   out,
              double *         in2,
              fftw_complex *   out2 );

#define USE_BATCHED 1
#define USE_CUFFTW 1

#if USE_BATCHED
void solver( System sys ) {

    PUSH_RANGE( "solver", 0 )

    DSTN dst;
    int  Nx = sys.lat.Nx, Ny = sys.lat.Ny;  //, Nxy = sys.lat.Nxy;

    double _Complex *rhat;
    double _Complex *xhat;
    CUDA_RT_CALL( cudaMallocHost( ( void ** )( &rhat ), sys.lat.Nxy * sizeof( double _Complex ) ) );
    CUDA_RT_CALL( cudaMallocHost( ( void ** )( &xhat ), sys.lat.Nxy * sizeof( double _Complex ) ) );

    int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;
    dst.Nx   = Nx;
    dst.N    = N;
    dst.coef = sqrt( 2.0 / ( Nx + 1 ) );

    size_t size_in  = sizeof( double ) * N * Ny;
    size_t size_out = sizeof( fftw_complex ) * NC * Ny;

#if USE_CUFFTW
    double *      in, *in2;
    fftw_complex *out, *out2;
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&in, size_in, 1 ) );
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&in2, size_in, 1 ) );
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&out, size_out, 1 ) );
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&out2, size_out, 1 ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( in, size_in, 0, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( in2, size_in, 0, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( out, size_out, 0, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( out2, size_out, 0, NULL ) );

    CUDA_RT_CALL( cudaMemset( in, size_in, 0 ) );
    CUDA_RT_CALL( cudaMemset( in2, size_in, 0 ) );
    CUDA_RT_CALL( cudaMemset( out, size_out, 0 ) );
    CUDA_RT_CALL( cudaMemset( out2, size_out, 0 ) );

#else
    double *      in   = ( double * )fftw_malloc( size_in );        /********************* FFTW *********************/
    double *      in2  = ( double * )fftw_malloc( size_in );        /********************* FFTW *********************/
    fftw_complex *out  = ( fftw_complex * )fftw_malloc( size_out ); /********************* FFTW *********************/
    fftw_complex *out2 = ( fftw_complex * )fftw_malloc( size_out ); /********************* FFTW *********************/
#endif

    cuDoubleComplex *d_rhat;
    cuDoubleComplex *d_xhat;
    cuDoubleComplex *d_y;

    CUDA_RT_CALL( cudaMalloc( ( void ** )( &d_rhat ), sys.lat.Nxy * sizeof( cuDoubleComplex ) ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )( &d_xhat ), sys.lat.Nxy * sizeof( cuDoubleComplex ) ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )( &d_y ), sys.lat.Nxy * sizeof( cuDoubleComplex ) ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.rhs, sys.lat.Nxy * sizeof( double _Complex ), 0, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.sol, sys.lat.Nxy * sizeof( double _Complex ), 0, NULL ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.U, sys.lat.Nxy * sizeof( double _Complex ), 0, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.L, sys.lat.Nxy * sizeof( double _Complex ), 0, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.Up, sys.lat.Nxy * sizeof( double _Complex ), 0, NULL ) );

    /**********************BATCHED***************************/
    int  rank    = 1; /* not 2: we are computing 1d transforms */
    int  n[]     = { N };
    int  howmany = Ny;
    int  idist   = N;
    int  odist   = NC;
    int  istride = 1;
    int  ostride = 1; /* distance between two elements in the same column */
    int *inembed = NULL;
    int *onembed = NULL;
    /**********************BATCHED***************************/

    fftw_plan plan, plan2; /********************* FFTW *********************/

    PUSH_RANGE( "1st fffw_plan", 1 )
    plan = fftw_plan_many_dft_r2c(
        rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_ESTIMATE );
    plan2 = fftw_plan_many_dft_r2c(
        rank, n, howmany, in2, inembed, istride, idist, out2, onembed, ostride, odist, FFTW_ESTIMATE );
    POP_RANGE

    PUSH_RANGE( "DST", 5 )
    fullDST( sys, dst, plan, plan2, d_rhat, d_xhat, d_y, in, out, in2, out2 );

    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.sol, sys.lat.Nxy * sizeof( double _Complex ), cudaCpuDeviceId, NULL ) );
    POP_RANGE

    PUSH_RANGE( "Cleanup", 6 )

#if USE_CUFFTW
    CUDA_RT_CALL( cudaFree( in ) );
    CUDA_RT_CALL( cudaFree( out ) );
    CUDA_RT_CALL( cudaFree( in2 ) );
    CUDA_RT_CALL( cudaFree( out2 ) );
#else
    free( in );
    in = NULL;
    fftw_free( out );
    out = NULL; /********************* FFTW *********************/
    free( in2 );
    in = NULL;
    fftw_free( out2 );
    out2 = NULL; /********************* FFTW *********************/
#endif

    CUDA_RT_CALL( cudaFree( d_rhat ) );
    CUDA_RT_CALL( cudaFree( d_xhat ) );

    CUDA_RT_CALL( cudaFreeHost( rhat ) );
    CUDA_RT_CALL( cudaFreeHost( xhat ) );

    fftw_destroy_plan( plan );  /********************* FFTW *********************/
    fftw_destroy_plan( plan2 ); /********************* FFTW *********************/
    POP_RANGE

    POP_RANGE
}

#else

void solver( System sys ) {

    PUSH_RANGE( "solver", 0 )

    DSTN             dst;
    int              i, j, mx;
    int              Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double _Complex *rhat = ( double _Complex * )malloc( Nxy * sizeof( double _Complex ) );
    double _Complex *xhat = ( double _Complex * )malloc( Nxy * sizeof( double _Complex ) );

    int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;
    dst.Nx   = Nx;
    dst.N    = N;
    dst.coef = sqrt( 2.0 / ( Nx + 1 ) );

#if USE_OMP
#pragma omp parallel private( i, j, mx )
    {
#endif

        size_t        size_in  = sizeof( double ) * N;
        size_t        size_out = sizeof( fftw_complex ) * NC;

#if USE_CUFFTW
        double *      in, *in2;
        fftw_complex *out, *out2;
        CUDA_RT_CALL( cudaMallocHost( ( void ** )&in, size_in ) );
        CUDA_RT_CALL( cudaMallocHost( ( void ** )&in2, size_in ) );
        CUDA_RT_CALL( cudaMallocHost( ( void ** )&out, size_out ) );
        CUDA_RT_CALL( cudaMallocHost( ( void ** )&out2, size_out ) );
#else
    double *      in   = ( double * )fftw_malloc( size_in );        /********************* FFTW *********************/
    double *      in2  = ( double * )fftw_malloc( size_in );        /********************* FFTW *********************/
    fftw_complex *out  = ( fftw_complex * )fftw_malloc( size_out ); /********************* FFTW *********************/
    fftw_complex *out2 = ( fftw_complex * )fftw_malloc( size_out ); /********************* FFTW *********************/
#endif

        memset( in, 0, size_in );
        memset( in2, 0, size_in );
        memset( out, 0, size_out );
        memset( out2, 0, size_out );

        double _Complex *y = ( double _Complex * )malloc( Ny * sizeof( double _Complex ) );
        fftw_plan        plan, plan2; /********************* FFTW *********************/

        PUSH_RANGE( "1st fffw_plan", 1 )
#if USE_OMP
#pragma omp critical( make_plan )
#endif
        {
            plan = fftw_plan_dft_r2c_1d( N, in, out, FFTW_ESTIMATE ); /********************* FFTW *********************/
            plan2 =
                fftw_plan_dft_r2c_1d( N, in2, out2, FFTW_ESTIMATE ); /********************* FFTW *********************/
        }
        POP_RANGE

        PUSH_RANGE( "forwardDST", 2 )
        forwardDST( sys, dst, sys.rhs, rhat, plan, in, out, plan2, in2, out2 );
        POP_RANGE

        PUSH_RANGE( "Middle stuff", 3 )
#if USE_OMP
#pragma omp for
#endif
        for ( i = 0; i < Nx; i++ ) {
            y[0] = rhat[i];
            mx   = i * Ny;
            for ( j = 1; j < Ny; j++ ) {
                y[j] = rhat[ind( i, j, Nx )] - sys.L[j + mx] * y[j - 1];
            }
            xhat[Ny - 1 + mx] = y[Ny - 1] / sys.U[Ny - 1 + mx];
            for ( j = Ny - 2; j >= 0; j-- ) {
                xhat[j + mx] = ( y[j] - sys.Up[j + mx] * xhat[j + 1 + mx] ) / sys.U[j + mx];
            }
        }
        POP_RANGE

        PUSH_RANGE( "reverseDST", 4 )
        reverseDST( sys, dst, xhat, sys.sol, plan, in, out, plan2, in2, out2 );
        POP_RANGE

        PUSH_RANGE( "Cleanup", 5 )
#if USE_CUFFTW
        CUDA_RT_CALL( cudaFreeHost( in ) );
        CUDA_RT_CALL( cudaFreeHost( out ) );
        CUDA_RT_CALL( cudaFreeHost( in2 ) );
        CUDA_RT_CALL( cudaFreeHost( out2 ) );
#else
    free( in );
    in = NULL;
    fftw_free( out );
    out = NULL; /********************* FFTW *********************/
    free( in2 );
    in = NULL;
    fftw_free( out2 );
    out2 = NULL; /********************* FFTW *********************/
#endif
        fftw_destroy_plan( plan );  /********************* FFTW *********************/
        fftw_destroy_plan( plan2 ); /********************* FFTW *********************/
        free( y );
        y = NULL;

#if USE_OMP
    }
#endif

    free( rhat );
    rhat = NULL;
    free( xhat );
    xhat = NULL;
    POP_RANGE

    POP_RANGE
}

#endif