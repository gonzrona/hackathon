#include "../headers/structs.h"

#include "cuda_helper.h"
#include "cuda_kernels.h"
#include <cuComplex.h>
#include <cufftw.h>

#if USE_CUFFTW
#ifdef USE_COMBINE
void fullDST( const System     sys,
              const DSTN       dst,
              const fftw_plan  plan,
              const fftw_plan  plan2,
              cuDoubleComplex *d_y,
              double *         in,
              fftw_complex *   out,
              double *         in2,
              fftw_complex *   out2 ) {

    PUSH_RANGE( "forwardDST", 2 )
    load_1st_DST_wrapper( sys, dst, sys.rhs, in, in2 );
    fftw_execute( plan );  /********************* FFTW *********************/
    fftw_execute( plan2 ); /********************* FFTW *********************/
    POP_RANGE

    PUSH_RANGE( "forwardDST", 3 )
    middle_stuff_ls_DST_wrapper( sys, dst, out, out2, in, in2, d_y );
    POP_RANGE

    PUSH_RANGE( "forwardDST", 4 )
    fftw_execute( plan );  /********************* FFTW *********************/
    fftw_execute( plan2 ); /********************* FFTW *********************/

    store_2st_DST_wrapper( sys, dst, out, out2, sys.sol );
    POP_RANGE
}
#else
void fullDST( const System     sys,
              const DSTN       dst,
              const fftw_plan  plan,
              const fftw_plan  plan2,
              cuDoubleComplex *d_rhat,
              cuDoubleComplex *d_xhat,
              cuDoubleComplex *d_y,
              double *         in,
              fftw_complex *   out,
              double *         in2,
              fftw_complex *   out2 ) {

    PUSH_RANGE( "forwardDST", 2 )
    load_1st_DST_wrapper( sys, dst, sys.rhs, in, in2 );
    fftw_execute( plan );  /********************* FFTW *********************/
    fftw_execute( plan2 ); /********************* FFTW *********************/
    store_1st_DST_wrapper( sys, dst, out, out2, d_rhat );
    POP_RANGE

    PUSH_RANGE( "forwardDST", 3 )
    middle_stuff_DST_wrapper( sys, d_rhat, d_xhat, d_y );
    POP_RANGE

    PUSH_RANGE( "forwardDST", 4 )
    load_2st_DST_wrapper( sys, dst, d_xhat, in, in2 );
    fftw_execute( plan );  /********************* FFTW *********************/
    fftw_execute( plan2 ); /********************* FFTW *********************/

    store_2st_DST_wrapper( sys, dst, out, out2, sys.sol );
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