#include "../headers/structs.h"

#include "cuda_helper.h"
#include "cuda_kernels.h"
#include <cuComplex.h>
#include <cufftw.h>

#if USE_CUFFTW
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
              fftw_complex *      out2 ) {

    PUSH_RANGE( "1st DST", 2 )
    CUDA_RT_CALL( cudaEventRecord( events[2], streams[2] ) );

    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[0], cudaEventWaitDefault ) );  // Wait for plan creation
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[1], cudaEventWaitDefault ) );  // Wait for plan2 creation
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[2], cudaEventWaitDefault ) );  // Wait for sys.rhs

    load_1st_DST_wrapper( streams[0], sys, dst, sys.rhs, in, in2 );

    CUDA_RT_CALL( cudaEventRecord( events[0], streams[0] ) );

    CUDA_RT_CALL( cudaStreamWaitEvent( streams[1], events[0], cudaEventWaitDefault ) );  // Wait for load_1st_DST
    CUDA_RT_CALL( cufftSetStream( plan, streams[0] ) );
    CUDA_RT_CALL( cufftSetWorkArea( plan, d_workspace[0] ) );
    CUDA_RT_CALL( cufftExecD2Z( plan, in, out ) );  // Running in streams[0]

    CUDA_RT_CALL( cufftSetStream( plan, streams[1] ) );
    CUDA_RT_CALL( cufftSetWorkArea( plan, d_workspace[1] ) );
    CUDA_RT_CALL( cufftExecD2Z( plan, in2, out2 ) );  // Running in streams[1]

    CUDA_RT_CALL( cudaEventRecord( events[0], streams[0] ) );
    CUDA_RT_CALL( cudaEventRecord( events[1], streams[1] ) );
    CUDA_RT_CALL( cudaEventRecord( events[2], streams[3] ) );
    POP_RANGE

    PUSH_RANGE( "Trig Solver", 3 )
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[0], cudaEventWaitDefault ) );  // Wait for plan execution
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[1], cudaEventWaitDefault ) );  // Wait for plan2 execution
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[2], cudaEventWaitDefault ) );  // Wait forsys.U, sys.L, sys.Up

    middle_stuff_ls_DST_wrapper( streams[0], sys, dst, out, out2, in, in2, d_y );

    CUDA_RT_CALL( cudaEventRecord( events[0], streams[0] ) );
    POP_RANGE

    PUSH_RANGE( "2nd DST", 4 )
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[1], events[0], cudaEventWaitDefault ) );  // Wait for middle_stuff_ls_DST
    CUDA_RT_CALL( cufftSetStream( plan, streams[0] ) );

    CUDA_RT_CALL( cufftSetWorkArea( plan, d_workspace[0] ) );
    CUDA_RT_CALL( cufftExecD2Z( plan, in, out ) );  // Running in streams[0]

    CUDA_RT_CALL( cufftSetStream( plan, streams[1] ) );
    CUDA_RT_CALL( cufftSetWorkArea( plan, d_workspace[1] ) );
    CUDA_RT_CALL( cufftExecD2Z( plan, in2, out2 ) );  // Running in streams[1]

    CUDA_RT_CALL( cudaEventRecord( events[0], streams[0] ) );
    CUDA_RT_CALL( cudaEventRecord( events[1], streams[1] ) );
    CUDA_RT_CALL( cudaEventRecord( events[2], streams[4] ) );

    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[0], cudaEventWaitDefault ) );  // Wait for plan execution
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[1], cudaEventWaitDefault ) );  // Wait for plan2 execution
    CUDA_RT_CALL( cudaStreamWaitEvent( streams[0], events[2], cudaEventWaitDefault ) );  // Wait for sys.sol
    store_2st_DST_wrapper( streams[0], sys, dst, out, out2, sys.sol );

    CUDA_RT_CALL( cudaStreamSynchronize( streams[0] ) );  // Wait for store_2st_DST
    POP_RANGE
}
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
              fftw_complex *      out2 ) {

    PUSH_RANGE( "1st DST", 2 )
    load_1st_DST_wrapper( NULL, sys, dst, sys.rhs, in, in2 );

    CUDA_RT_CALL( cufftExecD2Z( plan, in, out ) );    // Running in streams[0]
    CUDA_RT_CALL( cufftExecD2Z( plan, in2, out2 ) );  // Running in streams[1]

    store_1st_DST_wrapper( NULL, sys, dst, out, out2, d_rhat );
    POP_RANGE

    PUSH_RANGE( "Trig Solver", 3 )
    middle_stuff_DST_wrapper( NULL, sys, d_rhat, d_xhat, d_y );
    POP_RANGE

    PUSH_RANGE( "2nd DST", 4 )
    load_2st_DST_wrapper( NULL, sys, dst, d_xhat, in, in2 );

    CUDA_RT_CALL( cufftExecD2Z( plan, in, out ) );    // Running in streams[0]
    CUDA_RT_CALL( cufftExecD2Z( plan, in2, out2 ) );  // Running in streams[1]

    store_2st_DST_wrapper( NULL, sys, dst, out, out2, sys.sol );
    POP_RANGE
}
#endif
#else
void DST( DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out ) {

    int i;

    for ( i = 0; i < dst.N; i++ ) {
        in[i] = 0.0;
    }

    for ( i = 0; i < dst.Nx; i++ ) {
        in[i + 1] = creal( b[i] );
    }

    fftw_execute( plan );

    for ( i = 0; i < dst.Nx; i++ ) {
        bhat[i] = -cimag( out[i + 1] );
    }

    for ( i = 0; i < dst.Nx; i++ ) {
        in[i + 1] = cimag( b[i] );
    }

    fftw_execute( plan );

    for ( i = 0; i < dst.Nx; i++ ) {
        bhat[i] = dst.coef * ( bhat[i] - I * cimag( out[i + 1] ) );
    }
}
#endif