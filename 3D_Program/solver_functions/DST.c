#include <cuComplex.h>
#include <cufft.h>

#include "../headers/structs.h"

#include "cuda_helper.h"
#include "cuda_kernels.h"

// #define USE_CUFFTW 1

#ifdef USE_CUFFTW

void DST1( const cudaStream_t *streams,
           const int           l,
           const int           Nx,
           const int           Ny,
           double complex * rhs,
           cuDoubleComplex *rhat,
           cufftHandle      p1,
           double *         in1,
           cuDoubleComplex *out1,
           cufftHandle      p2,
           double *         in2,
           cuDoubleComplex *out2 ) {

    int Nxy = Nx * Ny;

    double coef;
    coef = 2.0 / sqrt( Nx + 1 ) / sqrt( Ny + 1 );

    int i, j, NR, NC;
    NR = 2 * Nx + 2;
    NC = NR / 2 + 1;

    load_1st_DST_wrapper( streams[0], l, Nx, Ny, NR, rhs, in1 );

    CUDA_RT_CALL( cufftExecD2Z( p1, in1, out1 ) );

    NR = 2 * Ny + 2;
    load_2st_DST_wrapper( streams[0], l, Nx, Ny, NR, NC, out1, in2 );

    NC = NR / 2 + 1;
    CUDA_RT_CALL( cufftExecD2Z( p2, in2, out2 ) );

    store_1st_DST_wrapper( streams[0], l, Nx, Ny, NR, NC, out2, rhat );

    return;
}

void DST2( const cudaStream_t *streams,
           const int           l,
           const int           Nx,
           const int           Ny,
           const int           Nz,
           double complex * xhat,
           cuDoubleComplex *sol,
           cufftHandle      p1,
           double *         in1,
           cuDoubleComplex *out1,
           cufftHandle      p2,
           double *         in2,
           cuDoubleComplex *out2 ) {

    int Nxy = Nx * Ny;

    double coef;
    coef = 2.0 / sqrt( Nx + 1 ) / sqrt( Ny + 1 );

    int i, j, NR, NC;
    NR = 2 * Nx + 2;
    NC = NR / 2 + 1;

    load_3st_DST_wrapper( streams[0], l, Nx, Ny, Nz, NR, xhat, in1 );

    CUDA_RT_CALL( cufftExecD2Z( p1, in1, out1 ) );

    NR = 2 * Ny + 2;
    load_4st_DST_wrapper( streams[0], l, Nx, Ny, NR, NC, out1, in2 );

    NC = NR / 2 + 1;
    CUDA_RT_CALL( cufftExecD2Z( p2, in2, out2 ) );

    store_2st_DST_wrapper( streams[0], l, Nx, Ny, NR, NC, out2, sol );

    return;
}

void DST( const int        Nx,
          const int        Ny,
          double *         b_2D,
          double *         bhat,
          cufftHandle      p1,
          double *         in1,
          cuDoubleComplex *out1,
          cufftHandle      p2,
          double *         in2,
          cuDoubleComplex *out2 ) {
    double coef;
    coef = 2.0 / sqrt( Nx + 1 ) / sqrt( Ny + 1 );

    int i, j, NR, NC;
    NR = 2 * Nx + 2;
    NC = NR / 2 + 1;

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            in1[i + 1 + j * NR] = b_2D[i + j * Nx];
        }
    }

    CUDA_RT_CALL( cufftExecD2Z( p1, in1, out1 ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    NR = 2 * Ny + 2;

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            b_2D[i + j * Nx] = out1[i + 1 + j * NC].y;
        }
    }

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            in2[j + 1 + i * NR] = b_2D[i + j * Nx];
        }
    }

    NC = NR / 2 + 1;

    CUDA_RT_CALL( cufftExecD2Z( p2, in2, out2 ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    for ( i = 0; i < Nx; i++ ) {
        for ( j = 0; j < Ny; j++ ) {
            bhat[i + j * Nx] = out2[j + 1 + i * NC].y;
        }
    }

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            bhat[i + j * Nx] = coef * bhat[i + j * Nx];
        }
    }

    return;
}
#else
void DST( const int     Nx,
          const int     Ny,
          double *      b_2D,
          double *      bhat,
          fftw_plan     p1,
          double *      in1,
          fftw_complex *out1,
          fftw_plan     p2,
          double *      in2,
          fftw_complex *out2 ) {
    double coef;
    coef = 2.0 / sqrt( Nx + 1 ) / sqrt( Ny + 1 );

    int i, j, NR, NC;
    NR = 2 * Nx + 2;
    NC = NR / 2 + 1;

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            in1[i + 1 + j * NR] = b_2D[i + j * Nx];
        }
    }

    fftw_execute( p1 );

    NR = 2 * Ny + 2;

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            b_2D[i + j * Nx] = cimag( out1[i + 1 + j * NC] );
        }
    }

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            in2[j + 1 + i * NR] = b_2D[i + j * Nx];
        }
    }

    NC = NR / 2 + 1;

    fftw_execute( p2 );

    for ( i = 0; i < Nx; i++ ) {
        for ( j = 0; j < Ny; j++ ) {
            bhat[i + j * Nx] = cimag( out2[j + 1 + i * NC] );
        }
    }

    for ( j = 0; j < Ny; j++ ) {
        for ( i = 0; i < Nx; i++ ) {
            bhat[i + j * Nx] = coef * bhat[i + j * Nx];
        }
    }

    return;
}
#endif
