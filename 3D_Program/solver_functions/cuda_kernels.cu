#include <stdio.h>

#include "cuda_helper.h"
#include "cuda_kernels.h"

__device__ cuDoubleComplex ComplexScale( cuDoubleComplex const &a, double const &scale ) {
    cuDoubleComplex c;
    c.x = a.x * scale;
    c.y = a.y * scale;
    return ( c );
}

// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         in1[i + 1 + j * NR] = creal( rhs[( j * Nx + i ) + ( l * Nxy )] );
//     }
// }
// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         in1[i + 1 + j * NR] = cimag( rhs[( j * Nx + i ) + ( l * Nxy )] );
//     }
// }
__global__ void __launch_bounds__( 256 ) load_1st_DST( const int l,
                                                       const int NR,
                                                       const int Nx,
                                                       const int Ny,
                                                       const cuDoubleComplex *__restrict__ rhs,
                                                       double *__restrict__ in1 ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            in1[tidY * NR + tidX + 1]                   = rhs[( l * Nx * Ny ) + ( tidY * Nx + tidX )].x;
            in1[( NR * Ny ) + ( tidY * NR + tidX + 1 )] = rhs[( l * Nx * Ny ) + ( tidY * Nx + tidX )].y;
        }
    }
}

// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         in2[j + 1 + i * NR] = out1[i + 1 + j * NC].y;
//     }
// }
__global__ void __launch_bounds__( 256 ) load_2st_DST( const int l,
                                                       const int NR,
                                                       const int NC,
                                                       const int Nx,
                                                       const int Ny,
                                                       const cuDoubleComplex *__restrict__ out1,
                                                       double *__restrict__ in2 ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            in2[tidX * NR + tidY + 1]                   = out1[tidY * NC + tidX + 1].y;
            in2[( NR * Nx ) + ( tidX * NR + tidY + 1 )] = out1[( NC * Ny ) + ( tidY * NC + tidX + 1 )].y;
        }
    }
}

// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         rhat[( j * Nx + i ) + ( l * Nxy )].x = coef * out2[j + 1 + i * NC].y;
//         rhat[( j * Nx + i ) + ( l * Nxy )].y = coef * out2[( NC * Ny ) + (j + 1 + i * NC)].y;
//     }
// }
__global__ void __launch_bounds__( 256 ) store_1st_DST( const int    l,
                                                        const int    NR,
                                                        const int    NC,
                                                        const int    Nx,
                                                        const int    Ny,
                                                        const double coef,
                                                        const cuDoubleComplex *__restrict__ out2,
                                                        cuDoubleComplex *__restrict__ d_rhat ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            d_rhat[( Nx * tidY + tidX ) + ( l * Nx * Ny )].x = coef * out2[tidX * NC + tidY + 1].y;
            d_rhat[( Nx * tidY + tidX ) + ( l * Nx * Ny )].y = coef * out2[( NC * Nx ) + ( tidX * NC + tidY + 1 )].y;
        }
    }
}

// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         in1[i + 1 + j * NR] = creal( rhs[( j * Nx + i ) + ( l * Nxy )] );
//     }
// }
__global__ void __launch_bounds__( 256 ) load_3st_DST( const int l,
                                                       const int NR,
                                                       const int Nx,
                                                       const int Ny,
                                                       const int Nz,
                                                       const cuDoubleComplex *__restrict__ d_xhat,
                                                       double *__restrict__ in1 ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            in1[tidY * NR + tidX + 1]                   = d_xhat[l + ( tidY * Nx + tidX ) * Nz].x;
            in1[( NR * Ny ) + ( tidY * NR + tidX + 1 )] = d_xhat[l + ( tidY * Nx + tidX ) * Nz].y;
        }
    }
}

// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         in2[j + 1 + i * NR] = out1[i + 1 + j * NC].y;
//     }
// }
__global__ void __launch_bounds__( 256 ) load_4st_DST( const int l,
                                                       const int NR,
                                                       const int NC,
                                                       const int Nx,
                                                       const int Ny,
                                                       const cuDoubleComplex *__restrict__ out1,
                                                       double *__restrict__ in2 ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            in2[tidX * NR + tidY + 1]                   = out1[tidY * NC + tidX + 1].y;
            in2[( NR * Nx ) + ( tidX * NR + tidY + 1 )] = out1[( NC * Ny ) + ( tidY * NC + tidX + 1 )].y;
        }
    }
}

// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         sol[( j * Nx + i ) + ( l * Nxy )].y = coef * out2[j + 1 + i * NC].y;
//     }
// }
__global__ void __launch_bounds__( 256 ) store_2st_DST( const int    l,
                                                        const int    NR,
                                                        const int    NC,
                                                        const int    Nx,
                                                        const int    Ny,
                                                        const double coef,
                                                        const cuDoubleComplex *__restrict__ out2,
                                                        cuDoubleComplex *__restrict__ d_sol ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            d_sol[( tidY * Nx + tidX ) + ( l * Nx * Ny )].x = coef * out2[tidX * NC + tidY + 1].y;
            d_sol[( tidY * Nx + tidX ) + ( l * Nx * Ny )].y = coef * out2[( NC * Nx ) + ( tidX * NC + tidY + 1 )].y;
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

// for ( j = 0; j < Ny; j++ ) {
//     for ( i = 0; i < Nx; i++ ) {
//         y_a[0] = rhat[i + j * Nx];

//         for ( l = 1; l < Nz; l++ ) {
//             y_a[l] = rhat[i + j * Nx + l * Nxy] - sys.L[l + i * Nz + j * Nxz] * y_a[l - 1];
//         }

//         xhat[Nz - 1 + i * Nz + j * Nxz] = y_a[Nz - 1] * sys.U[Nz - 1 + i * Nz + j * Nxz];

//         for ( l = Nz - 2; l >= 0; l-- ) {
//             xhat[l + i * Nz + j * Nxz] =
//                 ( y_a[l] - sys.Up[l + i * Nz + j * Nxz] * xhat[l + 1 + i * Nz + j * Nxz] ) *
//                 sys.U[l + i * Nz + j * Nxz];
//         }
//     }
// }

__global__ void __launch_bounds__( 256 ) triangular_solver( const int Nx,
                                                            const int Ny,
                                                            const int Nz,
                                                            const cuDoubleComplex *__restrict__ d_SysU,
                                                            const cuDoubleComplex *__restrict__ d_SysL,
                                                            const cuDoubleComplex *__restrict__ d_SysUp,
                                                            const cuDoubleComplex *__restrict__ d_rhat,
                                                            cuDoubleComplex *__restrict__ d_xhat,
                                                            cuDoubleComplex *__restrict__ d_y ) {
    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    const int Nxy = Nx * Ny;
    const int Nxz = Nx * Nz;

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {

            int idx_y = ( tidY * Nx + tidX ) * Nz;

            d_y[idx_y] = d_rhat[tidY * Nx + tidX];

            for ( int l = 1; l < Nz; l++ ) {
                d_y[idx_y + l] = cuCsub( d_rhat[Nxy * l + tidY * Nx + tidX],
                                         cuCmul( d_SysL[Nxz * tidY + Nz * tidX + l], d_y[idx_y + ( l - 1 )] ) );
            }

            d_xhat[Nxz * tidY + Nz * tidX + ( Nz - 1 )] =
                cuCmul( d_y[idx_y + ( Nz - 1 )], d_SysU[Nxz * tidY + Nz * tidX + ( Nz - 1 )] );

            for ( int l = Nz - 2; l >= 0; l-- ) {
                d_xhat[Nxz * tidY + Nz * tidX + l] = cuCmul(
                    cuCsub( d_y[idx_y + l],
                            cuCmul( d_SysUp[Nxz * tidY + Nz * tidX + l], d_xhat[Nxz * tidY + Nz * tidX + 1 + l] ) ),
                    d_SysU[Nxz * tidY + Nz * tidX + l] );
            }
        }
    }
}

// // #pragma omp for
// //   for (j = 0; j < Ny; j++) {
// //     my = j * Nx;

// //     for (i = 0; i < dst.Nx; i++) {
// //       rhat[i + my] = dst.coef * (-cimag(out[(j * NC) + i + 1]) -
// //                                  I * cimag(out2[(j * NC) + i + 1]));
// //     }
// //   }

// // for (i = 0; i < Nx; i++) {
// //   y[0] = rhat[i];
// //   mx = i * Ny;
// //   for (j = 1; j < Ny; j++) {
// //     y[j] = rhat[ind(i, j, Nx)] - sys.L[j + mx] * y[j - 1];
// //   }
// //   xhat[Ny - 1 + mx] = y[Ny - 1] / sys.U[Ny - 1 + mx];
// //   for (j = Ny - 2; j >= 0; j--) {
// //     xhat[j + mx] = (y[j] - sys.Up[j + mx] * xhat[j + 1 + mx]) / sys.U[j + mx];
// //   }
// // }

// // #pragma omp for
// //   for (j = 0; j < Ny; j++) {
// //     for (i = 0; i < dst.Nx; i++) {
// //       in[(j * N) + i + 1] = creal(xhat[j + i * Ny]);
// //     }
// //     for (i = 0; i < dst.Nx; i++) {
// //       in2[(j * N) + i + 1] = cimag(xhat[j + i * Ny]);
// //     }
// //   }
// __global__ void __launch_bounds__( 64 ) middle_stuff_ls_DST( const int    N,
//                                                              const int    Nx,
//                                                              const int    Ny,
//                                                              const int    NC,
//                                                              const double coef,
//                                                              const cuDoubleComplex *__restrict__ out,
//                                                              const cuDoubleComplex *__restrict__ d_SysU,
//                                                              const cuDoubleComplex *__restrict__ d_SysL,
//                                                              const cuDoubleComplex *__restrict__ d_SysUp,
//                                                              cuDoubleComplex *__restrict__ d_y,
//                                                              double *__restrict__ in ) {

//     const int tidX { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };

//     cuDoubleComplex temp {};

//     if ( tidX < Nx ) {
//         int mx = Ny * tidX;

//         temp      = make_cuDoubleComplex( -out[tidX + 1].y, -out[( NC * Ny ) + tidX + 1].y );
//         temp      = ComplexScale( temp, coef );
//         d_y[tidX] = temp;

// #pragma unroll 8
//         for ( int j = 1; j < Ny; j++ ) {
//             cuDoubleComplex temp2 = cuCmul( d_SysL[mx + j], d_y[( j - 1 ) * Ny + tidX] );
//             temp = make_cuDoubleComplex( -out[j * NC + tidX + 1].y, -out[( NC * Ny ) + ( j * NC + tidX + 1 )].y );
//             temp = ComplexScale( temp, coef );
//             d_y[j * Ny + tidX] = cuCsub( temp, temp2 );
//         }

//         temp = cuCdiv( d_y[( Ny - 1 ) * Ny + tidX], d_SysU[mx + ( Ny - 1 )] );

//         in[( Ny - 1 ) * N + tidX + 1]                  = temp.x;
//         in[( N * Ny ) + ( ( Ny - 1 ) * N + tidX + 1 )] = temp.y;
// #pragma unroll 4
//         for ( int j = Ny - 2; j >= 0; j-- ) {
//             cuDoubleComplex temp2 =
//                 cuCdiv( cuCsub( d_y[j * Ny + tidX], cuCmul( d_SysUp[mx + j], temp ) ), d_SysU[mx + j] );
//             in[j * N + tidX + 1]                  = temp2.x;
//             in[( N * Ny ) + ( j * N + tidX + 1 )] = temp2.y;
//             temp                                  = temp2;
//         }
//     }
// }

void load_1st_DST_wrapper( const cudaStream_t stream,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                NR,
                           cuDoubleComplex *  d_rhs,
                           double *           in ) {

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &l, &NR, &Nx, &Ny, &d_rhs, &in };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &load_1st_DST ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}

void load_2st_DST_wrapper( const cudaStream_t stream,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                NR,
                           int                NC,
                           cuDoubleComplex *  d_rhs,
                           double *           in ) {

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &l, &NR, &NC, &Nx, &Ny, &d_rhs, &in };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &load_2st_DST ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}

void store_1st_DST_wrapper( const cudaStream_t stream,
                            int                l,
                            int                Nx,
                            int                Ny,
                            int                NR,
                            int                NC,
                            cuDoubleComplex *  out,
                            cuDoubleComplex *  d_rhat ) {

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    double coef = 2.0 / sqrt( Nx + 1 ) / sqrt( Ny + 1 );

    void *args[] { &l, &NR, &NC, &Nx, &Ny, &coef, &out, &d_rhat };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &store_1st_DST ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}

void load_3st_DST_wrapper( const cudaStream_t stream,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                Nz,
                           int                NR,
                           cuDoubleComplex *  d_rhs,
                           double *           in ) {

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &l, &NR, &Nx, &Ny, &Nz, &d_rhs, &in };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &load_3st_DST ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}

void load_4st_DST_wrapper( const cudaStream_t stream,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                NR,
                           int                NC,
                           cuDoubleComplex *  d_rhs,
                           double *           in ) {

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &l, &NR, &NC, &Nx, &Ny, &d_rhs, &in };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &load_4st_DST ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}

void store_2st_DST_wrapper( const cudaStream_t stream,
                            int                l,
                            int                Nx,
                            int                Ny,
                            int                NR,
                            int                NC,
                            cuDoubleComplex *  out,
                            cuDoubleComplex *  d_rhat ) {

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    double coef = 2.0 / sqrt( Nx + 1 ) / sqrt( Ny + 1 );

    void *args[] { &l, &NR, &NC, &Nx, &Ny, &coef, &out, &d_rhat };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &store_2st_DST ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}

void triangular_solver_wrapper( const cudaStream_t     stream,
                                System                 sys,
                                int                    Nx,
                                int                    Ny,
                                int                    Nz,
                                const cuDoubleComplex *d_rhat,
                                cuDoubleComplex *      d_xhat,
                                cuDoubleComplex *      d_y ) {

    // int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    // int N = 2 * Nx + 2;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &Nx, &Ny, &Nz, &sys.U, &sys.L, &sys.Up, &d_rhat, &d_xhat, &d_y };

    CUDA_RT_CALL(
        cudaLaunchKernel( ( void * )( &triangular_solver ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}

// void middle_stuff_ls_DST_wrapper( const cudaStream_t stream,
//                                   System             sys,
//                                   //   const DSTN             dst,
//                                   const cuDoubleComplex *out,
//                                   double *               in,
//                                   cuDoubleComplex *      d_y ) {

//     int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
//     int N = 2 * Nx + 2, NC = ( N / 2 ) + 1;

//     double coef = 2.0 / sqrt( Nx + 1 ) / sqrt( Ny + 1 );

//     int numSMs;
//     CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

//     int threadPerBlock { 64 };
//     int blocksPerGrid { ( N + threadPerBlock - 1 ) / threadPerBlock };

//     void *args[] { &N, &Nx, &Ny, &NC, &coef, &out, &sys.U, &sys.L, &sys.Up, &d_y, &in };

//     CUDA_RT_CALL(
//         cudaLaunchKernel( ( void * )( &middle_stuff_ls_DST ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

//     CUDA_RT_CALL( cudaPeekAtLastError( ) );
// }