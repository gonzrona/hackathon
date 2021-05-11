#include "../headers/structs.h"

#include <assert.h>
#include <string.h>

#include <cuComplex.h>
#include <cufftw.h>

#include "cuda_helper.h"
#include "cuda_kernels.h"

void DST( DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out );

#ifdef USE_COMBINE
void fullDST( const cudaStream_t *streams,
              const cudaEvent_t * events,
              const System        sys,
              const DSTN          dst,
              const cufftHandle   plan,
              cuDoubleComplex *   d_workspace[2],
              cuDoubleComplex *   d_y,
              double *            in,
              fftw_complex *      out,
              double *            in2,
              fftw_complex *      out2 );
#else
void fullDST( const cudaStream_t *streams,
              const cudaEvent_t * events,
              const System        sys,
              const DSTN          dst,
              const cufftHandle   plan,
              cuDoubleComplex *   d_rhat,
              cuDoubleComplex *   d_xhat,
              cuDoubleComplex *   d_y,
              double *            in,
              fftw_complex *      out,
              double *            in2,
              fftw_complex *      out2 );
#endif

#if USE_CUFFTW
void solver( System sys ) {

    PUSH_RANGE( "solver", 0 )

    DSTN dst;
    int  Nx = sys.lat.Nx, Ny = sys.lat.Ny;  //, Nxy = sys.lat.Nxy;

    int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;
    dst.Nx   = Nx;
    dst.N    = N;
    dst.coef = sqrt( 2.0 / ( Nx + 1 ) );

    size_t size_in  = sizeof( double ) * N * Ny;
    size_t size_out = sizeof( fftw_complex ) * NC * Ny;

    double *      in, *in2;
    fftw_complex *out, *out2;

    CUDA_RT_CALL( cudaMalloc( ( void ** )( &in ), size_in ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )( &in2 ), size_in ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )( &out ), size_out ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )( &out2 ), size_out ) );

    PUSH_RANGE( "stream creation", 7 )
    int          num_streams = 5;
    cudaStream_t streams[num_streams];
    for ( int i = 0; i < num_streams; i++ ) {
        CUDA_RT_CALL( cudaStreamCreateWithFlags( &streams[i], cudaStreamNonBlocking ) );
    }
    POP_RANGE

    PUSH_RANGE( "stream creation", 7 )
    int         num_events = 5;
    cudaEvent_t events[num_events];
    for ( int i = 0; i < num_events; i++ ) {
        CUDA_RT_CALL( cudaEventCreateWithFlags( &events[i], cudaEventDisableTiming ) );
    }
    POP_RANGE

    cuDoubleComplex *d_y;
    CUDA_RT_CALL( cudaMalloc( ( void ** )( &d_y ), sys.lat.Nxy * sizeof( cuDoubleComplex ) ) );

#ifndef USE_COMBINE
    cuDoubleComplex *d_rhat;
    cuDoubleComplex *d_xhat;

    CUDA_RT_CALL( cudaMalloc( ( void ** )( &d_rhat ), sys.lat.Nxy * sizeof( cuDoubleComplex ) ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )( &d_xhat ), sys.lat.Nxy * sizeof( cuDoubleComplex ) ) );
#endif

    CUDA_RT_CALL( cudaMemsetAsync( in, size_in, 0, NULL ) );
    CUDA_RT_CALL( cudaMemsetAsync( in2, size_in, 0, NULL ) );
    CUDA_RT_CALL( cudaMemsetAsync( out, size_out, 0, NULL ) );
    CUDA_RT_CALL( cudaMemsetAsync( out2, size_out, 0, NULL ) );

    // The code should be here, BUT there's a bug in cuFFT
    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.rhs, sys.lat.Nxy * sizeof( double _Complex ), 0, streams[2] ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.U, sys.lat.Nxy * sizeof( double _Complex ), 0, streams[3] ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.L, sys.lat.Nxy * sizeof( double _Complex ), 0, streams[3] ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.Up, sys.lat.Nxy * sizeof( double _Complex ), 0, streams[3] ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( sys.sol, sys.lat.Nxy * sizeof( double _Complex ), 0, streams[4] ) );

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

    PUSH_RANGE( "cufft creation", 7 )
#ifdef USE_COMBINE
    cufftHandle      plan;
    size_t           workspace;
    cuDoubleComplex *d_workspace[2];

    CUDA_RT_CALL( cufftCreate( &plan ) );
    CUDA_RT_CALL( cufftSetAutoAllocation( plan, 0 ) );
    CUDA_RT_CALL( cufftMakePlanMany(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, howmany, &workspace ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )&d_workspace[0], workspace ) );
    CUDA_RT_CALL( cudaMalloc( ( void ** )&d_workspace[1], workspace ) );
#else
    cufftHandle     plan;
    size_t          workspace;
    cuDoubleComplex d_workspace;

    CUDA_RT_CALL( cufftCreate( &plan ) );
    CUDA_RT_CALL( cufftMakePlanMany(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, howmany, &workspace ) );

#endif
    POP_RANGE

    PUSH_RANGE( "DST", 5 )
#ifdef USE_COMBINE
    fullDST( streams, events, sys, dst, plan, d_workspace, d_y, in, out, in2, out2 );
#else
    fullDST( streams, events, sys, dst, plan, d_rhat, d_xhat, d_y, in, out, in2, out2 );
#endif

    CUDA_RT_CALL(
        cudaMemPrefetchAsync( sys.sol, sys.lat.Nxy * sizeof( double _Complex ), cudaCpuDeviceId, streams[0] ) );
    POP_RANGE

    PUSH_RANGE( "Cleanup", 6 )

    CUDA_RT_CALL( cudaFree( in ) );
    CUDA_RT_CALL( cudaFree( out ) );
    CUDA_RT_CALL( cudaFree( in2 ) );
    CUDA_RT_CALL( cudaFree( out2 ) );

    for ( int i = 0; i < num_streams; i++ ) {
        CUDA_RT_CALL( cudaStreamDestroy( streams[i] ) );
    }

#ifndef USE_COMBINE
    CUDA_RT_CALL( cudaFree( d_rhat ) );
    CUDA_RT_CALL( cudaFree( d_xhat ) );
#endif

    // for ( int i = 0; i < 2; i++ ) {
    CUDA_RT_CALL( cufftDestroy( plan ) ); /********************* FFTW *********************/
    // }
    POP_RANGE

    POP_RANGE
}

#else

void solver( System sys ) {

    PUSH_RANGE( "solver", 0 )

    DSTN             dst;
    int              i, j, my, mx;
    int              Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double _Complex *rhat = ( double _Complex * )malloc( Nxy * sizeof( double _Complex ) );
    double _Complex *xhat = ( double _Complex * )malloc( Nxy * sizeof( double _Complex ) );

    int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;
    dst.Nx   = Nx;
    dst.N    = N;
    dst.coef = sqrt( 2.0 / ( Nx + 1 ) );

#pragma omp parallel private( i, j, mx, my )
    {
        double *      in  = ( double * )fftw_malloc( sizeof( double ) * N );
        fftw_complex *out = ( fftw_complex * )fftw_malloc( sizeof( fftw_complex ) * NC );

        double _Complex *b    = ( double _Complex * )malloc( Nx * sizeof( double _Complex ) );
        double _Complex *bhat = ( double _Complex * )malloc( Nx * sizeof( double _Complex ) );
        double _Complex *y    = ( double _Complex * )malloc( Ny * sizeof( double _Complex ) );
        fftw_plan        plan;

        PUSH_RANGE( "1st fffw_plan", 1 )
#pragma omp critical( make_plan )
        {
            plan = fftw_plan_dft_r2c_1d( N, in, out, FFTW_ESTIMATE );
        }
        POP_RANGE

        PUSH_RANGE( "DST", 5 )

        PUSH_RANGE( "forwardDST", 2 )
#pragma omp for
        for ( j = 0; j < Ny; j++ ) {
            my = j * Nx;
            for ( i = 0; i < Nx; i++ ) {
                b[i] = sys.rhs[i + my];
            }
            DST( dst, b, bhat, plan, in, out );
            for ( i = 0; i < Nx; i++ ) {
                rhat[i + my] = bhat[i];
            }
        }
        POP_RANGE

        PUSH_RANGE( "Middle stuff", 3 )
#pragma omp for
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
#pragma omp for
        for ( j = 0; j < Ny; j++ ) {
            my = j * Nx;
            for ( i = 0; i < Nx; i++ ) {
                b[i] = xhat[j + i * Ny];
            }
            DST( dst, b, bhat, plan, in, out );
            for ( i = 0; i < Nx; i++ ) {
                sys.sol[i + my] = bhat[i];
            }
        }
        POP_RANGE

        POP_RANGE

        PUSH_RANGE( "Cleanup", 6 )
        fftw_destroy_plan( plan );
        free( in );
        in = NULL;
        fftw_free( out );
        out = NULL;
        free( b );
        b = NULL;
        free( bhat );
        bhat = NULL;
        free( y );
        y = NULL;
    }

    free( rhat );
    rhat = NULL;
    free( xhat );
    xhat = NULL;
    POP_RANGE

    POP_RANGE
}

#endif