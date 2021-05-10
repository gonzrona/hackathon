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
void forwardDST( System           sys,
                 DSTN             dst,
                 double _Complex *rhs,
                 double _Complex *rhat,
                 fftw_plan        plan,
                 double *         in,
                 fftw_complex *   out,
                 fftw_plan        plan2,
                 double *         in2,
                 fftw_complex *   out2 ) {

    int i, j, my;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;

#pragma omp for
    for ( j = 0; j < Ny; j++ ) {
        my = j * Nx;

        for ( i = 0; i < dst.Nx; i++ ) {
            in[i + 1] = creal( rhs[i + my] );
        }
        for ( i = 0; i < dst.Nx; i++ ) {
            in2[i + 1] = cimag( rhs[i + my] );
        }

        fftw_execute( plan );  /********************* FFTW *********************/
        fftw_execute( plan2 ); /********************* FFTW *********************/

        for ( i = 0; i < dst.Nx; i++ ) {
            rhat[i + my] = dst.coef * ( -cimag( out[i + 1] ) - I * cimag( out2[i + 1] ) );
        }
    }
}

void reverseDST( System           sys,
                 DSTN             dst,
                 double _Complex *xhat,
                 double _Complex *sol,
                 fftw_plan        plan,
                 double *         in,
                 fftw_complex *   out,
                 fftw_plan        plan2,
                 double *         in2,
                 fftw_complex *   out2 ) {

    int i, j, my;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;

#pragma omp for
    for ( j = 0; j < Ny; j++ ) {
        my = j * Nx;

        for ( i = 0; i < dst.Nx; i++ ) {
            in[i + 1] = creal( xhat[j + i * Ny] );
        }
        for ( i = 0; i < dst.Nx; i++ ) {
            in2[i + 1] = cimag( xhat[j + i * Ny] );
        }

        fftw_execute( plan );  /********************* FFTW *********************/
        fftw_execute( plan2 ); /********************* FFTW *********************/

        for ( i = 0; i < dst.Nx; i++ ) {
            sol[i + my] = dst.coef * ( -cimag( out[i + 1] ) - I * cimag( out2[i + 1] ) );
        }
    }
}
#endif