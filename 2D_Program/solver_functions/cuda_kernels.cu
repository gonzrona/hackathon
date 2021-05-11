#include <stdio.h>

#include "cuda_helper.h"
#include "cuda_kernels.h"

__device__ cuDoubleComplex ComplexScale( cuDoubleComplex const &a, double const &scale ) {
    cuDoubleComplex c;
    c.x = a.x * scale;
    c.y = a.y * scale;
    return ( c );
}

// #pragma omp for
//     for(j = 0; j < Ny; j++) {
//         for (i=0; i<dst.Nx; i++) {
//            in[(j * N) + i + 1] = sys.rhs[i + j * Nx].x;
//         }
//         for (i=0; i<dst.Nx; i++) {
//            in2[(j * N) + i + 1] = sys.rhs[i + j * Nx].y;
//         }
//     }
__global__ void __launch_bounds__( 256 ) load_1st_DST( const int N,
                                                       const int Nx,
                                                       const int Ny,
                                                       const cuDoubleComplex *__restrict__ rhs,
                                                       double *__restrict__ in,
                                                       double *__restrict__ in2 ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            cuDoubleComplex temp = rhs[tidX + tidY * Nx];
            in[tidY * N + tidX + 1]  = temp.x;
            in2[tidY * N + tidX + 1] = temp.y;
        }
    }
}

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     my = j * Nx;

//     for (i = 0; i < dst.Nx; i++) {
//       rhat[i + my] = dst.coef * (-cimag(out[(j * NC) + i + 1]) -
//                                  I * cimag(out2[(j * NC) + i + 1]));
//     }
//   }
__global__ void __launch_bounds__( 256 ) store_1st_DST( const int    N,
                                                        const int    Nx,
                                                        const int    Ny,
                                                        const int    NC,
                                                        const double coef,
                                                        const cuDoubleComplex *__restrict__ out,
                                                        const cuDoubleComplex *__restrict__ out2,
                                                        cuDoubleComplex *__restrict__ d_rhat ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            d_rhat[Nx * tidY + tidX].x = coef * -out[tidY * NC + tidX + 1].y;
            d_rhat[Nx * tidY + tidX].y = coef * -out2[tidY * NC + tidX + 1].y;
        }
    }
}

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     for (i = 0; i < dst.Nx; i++) {
//       in[(j * N) + i + 1] = creal(xhat[j + i * Ny]);
//     }
//     for (i = 0; i < dst.Nx; i++) {
//       in2[(j * N) + i + 1] = cimag(xhat[j + i * Ny]);
//     }
//   }
__global__ void __launch_bounds__( 256 ) load_2st_DST( const int N,
                                                       const int Nx,
                                                       const int Ny,
                                                       const cuDoubleComplex *__restrict__ xhat,
                                                       double *__restrict__ in,
                                                       double *__restrict__ in2 ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            cuDoubleComplex temp = xhat[tidY + tidX * Ny];
            in[tidY * N + tidX + 1]  = temp.x;
            in2[tidY * N + tidX + 1] = temp.y;
        }
    }
}

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     my = j * Nx;

//     for (i = 0; i < dst.Nx; i++) {
//       sol[i + my] = dst.coef * (-cimag(out[(j * NC) + i + 1]) -
//                                 I * cimag(out2[(j * NC) + i + 1]));
//     }
//   }
__global__ void __launch_bounds__( 256 ) store_2st_DST( const int    N,
                                                        const int    Nx,
                                                        const int    Ny,
                                                        const int    NC,
                                                        const double coef,
                                                        const cuDoubleComplex *__restrict__ out,
                                                        const cuDoubleComplex *__restrict__ out2,
                                                        cuDoubleComplex *__restrict__ d_sol ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            d_sol[Nx * tidY + tidX].x = coef * -out[tidY * NC + tidX + 1].y;
            d_sol[Nx * tidY + tidX].y = coef * -out2[tidY * NC + tidX + 1].y;
        }
    }
}

// for (i = 0; i < Nx; i++) {
//   y[0] = rhat[i];
//   mx = i * Ny;
//   for (j = 1; j < Ny; j++) {
//     y[j] = rhat[ind(i, j, Nx)] - sys.L[j + mx] * y[j - 1];
//   }
//   xhat[Ny - 1 + mx] = y[Ny - 1] / sys.U[Ny - 1 + mx];
//   for (j = Ny - 2; j >= 0; j--) {
//     xhat[j + mx] = (y[j] - sys.Up[j + mx] * xhat[j + 1 + mx]) / sys.U[j + mx];
//   }
// }
__global__ void __launch_bounds__( 256 ) middle_stuff_DST( const int N,
                                                           const int Nx,
                                                           const int Ny,
                                                           const cuDoubleComplex *__restrict__ d_SysU,
                                                           const cuDoubleComplex *__restrict__ d_SysL,
                                                           const cuDoubleComplex *__restrict__ d_SysUp,
                                                           const cuDoubleComplex *__restrict__ d_rhat,
                                                           cuDoubleComplex *__restrict__ d_xhat,
                                                           cuDoubleComplex *__restrict__ d_y ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
        int mx = tidX * Ny;

        d_y[tidX] = d_rhat[tidX];

        for ( int j = 1; j < Ny; j++ ) {
            d_y[j * Ny + tidX] =
                // d_rhat[ind(tidX, j, Nx)] - sys.L[j + mx] * d_y[(j - 1) + mx];
                cuCsub( d_rhat[ind( tidX, j, Nx )], cuCmul( d_SysL[j + mx], d_y[( j - 1 ) * Ny + tidX] ) );
        }

        d_xhat[Ny - 1 + mx] = cuCdiv( d_y[( Ny - 1 ) * Ny + tidX], d_SysU[Ny - 1 + mx] );
        for ( int j = Ny - 2; j >= 0; j-- ) {
            d_xhat[j + mx] =
                // (d_y[j + mx] - sys.Up[j + mx] * d_xhat[j + 1 + mx]) / sys.U[j +
                // mx];
                cuCdiv( cuCsub( d_y[j * Ny + tidX], cuCmul( d_SysUp[j + mx], d_xhat[j + 1 + mx] ) ), d_SysU[j + mx] );
        }
    }
}

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     my = j * Nx;

//     for (i = 0; i < dst.Nx; i++) {
//       rhat[i + my] = dst.coef * (-cimag(out[(j * NC) + i + 1]) -
//                                  I * cimag(out2[(j * NC) + i + 1]));
//     }
//   }

// for (i = 0; i < Nx; i++) {
//   y[0] = rhat[i];
//   mx = i * Ny;
//   for (j = 1; j < Ny; j++) {
//     y[j] = rhat[ind(i, j, Nx)] - sys.L[j + mx] * y[j - 1];
//   }
//   xhat[Ny - 1 + mx] = y[Ny - 1] / sys.U[Ny - 1 + mx];
//   for (j = Ny - 2; j >= 0; j--) {
//     xhat[j + mx] = (y[j] - sys.Up[j + mx] * xhat[j + 1 + mx]) / sys.U[j + mx];
//   }
// }

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     for (i = 0; i < dst.Nx; i++) {
//       in[(j * N) + i + 1] = creal(xhat[j + i * Ny]);
//     }
//     for (i = 0; i < dst.Nx; i++) {
//       in2[(j * N) + i + 1] = cimag(xhat[j + i * Ny]);
//     }
//   }
__global__ void __launch_bounds__( 256 ) middle_stuff_ls_DST( const int    N,
                                                              const int    Nx,
                                                              const int    Ny,
                                                              const int    NC,
                                                              const double coef,
                                                              const cuDoubleComplex *__restrict__ out,
                                                              const cuDoubleComplex *__restrict__ out2,
                                                              const cuDoubleComplex *__restrict__ d_SysU,
                                                              const cuDoubleComplex *__restrict__ d_SysL,
                                                              const cuDoubleComplex *__restrict__ d_SysUp,
                                                              cuDoubleComplex *__restrict__ d_y,
                                                              double *__restrict__ in,
                                                              double *__restrict__ in2 ) {

    const int tidX { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };

    cuDoubleComplex temp {};

    if (tidX < Nx) {
        int mx = Ny * tidX;

        temp      = make_cuDoubleComplex( -out[tidX + 1].y, -out2[tidX + 1].y );
        temp      = ComplexScale( temp, coef );
        d_y[tidX] = temp;

#pragma unroll 8
        for ( int j = 1; j < Ny; j++ ) {
            cuDoubleComplex temp2 = cuCmul( d_SysL[mx + j], d_y[( j - 1 ) * Ny + tidX] );
            temp                  = make_cuDoubleComplex( -out[j * NC + tidX + 1].y, -out2[j * NC + tidX + 1].y );
            temp                  = ComplexScale( temp, coef );
            d_y[j * Ny + tidX]    = cuCsub( temp, temp2 );
        }

        temp = cuCdiv( d_y[( Ny - 1 ) * Ny + tidX], d_SysU[mx + ( Ny - 1 )] );

        in[( Ny - 1 ) * N + tidX + 1]  = temp.x;
        in2[( Ny - 1 ) * N + tidX + 1] = temp.y;
#pragma unroll 8
        for ( int j = Ny - 2; j >= 0; j-- ) {
            cuDoubleComplex temp2 =
                cuCdiv( cuCsub( d_y[j * Ny + tidX], cuCmul( d_SysUp[mx + j], temp ) ), d_SysU[mx + j] );
            in[j * N + tidX + 1]  = temp2.x;
            in2[j * N + tidX + 1] = temp2.y;
            temp                  = temp2;
        }
    }
}

void load_1st_DST_wrapper( const System sys, const DSTN dst, const cuDoubleComplex *d_rhs, double *in, double *in2 ) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2 * Nx + 2;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &N, &Nx, &Ny, &d_rhs, &in, &in2 };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &load_1st_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaStreamSynchronize( NULL ) );
}

void store_1st_DST_wrapper( const System           sys,
                            const DSTN             dst,
                            const cuDoubleComplex *out,
                            const cuDoubleComplex *out2,
                            cuDoubleComplex *      d_rhat ) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    double coef = dst.coef;

    void *args[] { &N, &Nx, &Ny, &NC, &coef, &out, &out2, &d_rhat };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &store_1st_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaStreamSynchronize( NULL ) );
}

void load_2st_DST_wrapper( const System sys, const DSTN dst, const cuDoubleComplex *d_xhat, double *in, double *in2 ) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2 * Nx + 2;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &N, &Nx, &Ny, &d_xhat, &in, &in2 };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &load_2st_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaStreamSynchronize( NULL ) );
}

void store_2st_DST_wrapper( const System           sys,
                            const DSTN             dst,
                            const cuDoubleComplex *out,
                            const cuDoubleComplex *out2,
                            cuDoubleComplex *      d_sol ) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    double coef = dst.coef;

    void *args[] { &N, &Nx, &Ny, &NC, &coef, &out, &out2, &d_sol };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &store_2st_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaStreamSynchronize( NULL ) );
}

void middle_stuff_DST_wrapper( System                 sys,
                               const cuDoubleComplex *d_rhat,
                               cuDoubleComplex *      d_xhat,
                               cuDoubleComplex *      d_y ) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2 * Nx + 2;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    int threadPerBlock { 64 };
    int blocksPerGrid( numSMs );

    void *args[] { &N, &Nx, &Ny, &sys.U, &sys.L, &sys.Up, &d_rhat, &d_xhat, &d_y };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &middle_stuff_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaStreamSynchronize( NULL ) );
}

void middle_stuff_ls_DST_wrapper( System                 sys,
                                  const DSTN             dst,
                                  const cuDoubleComplex *out,
                                  const cuDoubleComplex *out2,
                                  double *               in,
                                  double *               in2,
                                  cuDoubleComplex *      d_y ) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;

    double coef = dst.coef;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    int threadPerBlock { 64 };
    int blocksPerGrid { (N + threadPerBlock - 1) / threadPerBlock };

    void *args[] { &N, &Nx, &Ny, &NC, &coef, &out, &out2, &sys.U, &sys.L, &sys.Up, &d_y, &in, &in2 };

    CUDA_RT_CALL(
        cudaLaunchKernel( ( void * )( &middle_stuff_ls_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaStreamSynchronize( NULL ) );
}