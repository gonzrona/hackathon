// #include <complex.h>

#include <cuComplex.h>

#include <cuda/std/complex>

#include "../solver_functions/cuda_helper.h"

#include "lu_2nd_kernels.h"

// #define CMPLX( x, y ) __builtin_complex( ( double )( x ), ( double )( y ) )

__device__ cuda::std::complex<double> find_k2( const System sys, const int l ) {
    double z   = sys.lat.z0 + sys.lat.hz * l;
    double ans = sys.A - sys.B * sin( sys.C * z );
    return ( ans * ans );
}

// __device__ cuda::std::complex<double>  find_a(const System sys, const int l ) {
// 	return (  );
// }

__device__ cuda::std::complex<double> find_b( const System sys, const int l ) {
    double hx  = sys.lat.hx;
    double hz  = sys.lat.hz;
    double Rzx = hz * hz / hx / hx;
    return ( Rzx );
}

__device__ cuda::std::complex<double> find_c( const System sys, const int l ) {
    double hy  = sys.lat.hy;
    double hz  = sys.lat.hz;
    double Rzy = hz * hz / hy / hy;
    return ( Rzy );
}

__device__ cuda::std::complex<double> find_d( const System sys, const int l ) {
    double hx  = sys.lat.hx;
    double hy  = sys.lat.hy;
    double hz  = sys.lat.hz;
    double hz2 = hz * hz;
    double Rzx = hz2 / hx / hx;
    double Rzy = hz2 / hy / hy;
    return ( -2.0 * ( Rzx + Rzy + 1.0 ) + hz2 * find_k2( sys, l + 1 ) );
}

// __device__ cuda::std::complex<double>  find_ap(const System sys, const int l ) {
// 	return (  );
// }

// __device__ cuda::std::complex<double>  find_bp(const System sys, const int l ) {
// 	return (  );
// }

// __device__ cuda::std::complex<double>  find_cp(const System sys, const int l ) {
// 	return (  );
// }

__device__ cuda::std::complex<double> find_dp( const System sys, const int l ) {
    return ( 1.0 );
}

// __device__ cuda::std::complex<double>  find_am(const System sys, const int l ) {
// 	return (  );
// }

// __device__ cuda::std::complex<double>  find_bm(const System sys, const int l ) {
// 	return (  );
// }

// __device__ cuda::std::complex<double>  find_cm(const System sys, const int l ) {
// 	return (  );
// }

__device__ cuda::std::complex<double> find_dm( const System sys, const int l ) {
    return ( 1.0 );
}

__device__ double find_evx( const System sys, const int i ) {
    return ( 2.0 * cos( ( i + 1 ) * M_PI / ( sys.lat.Nx + 1 ) ) );
}

__device__ double find_evy( const System sys, const int j ) {
    return ( 2.0 * cos( ( j + 1 ) * M_PI / ( sys.lat.Ny + 1 ) ) );
}

__device__ cuda::std::complex<double> find_ev3D( const System sys, const int i, const int j, const int l ) {
    double ev_x  = find_evx( sys, i );
    double ev_y  = find_evy( sys, j );
    double ev_xy = ev_x * ev_y;

    // double ans   = find_a( sys, l ) * e_xy + find_b( sys, l ) * ev_x + find_c( sys, l ) * ev_y + find_d( sys, i );
    cuda::std::complex<double> ans = find_b( sys, l ) * ev_x + find_c( sys, l ) * ev_y + find_d( sys, l );
    return ( ans );
}

__device__ cuda::std::complex<double> find_ev3DP( const System sys, const int i, const int j, const int l ) {
    // double ev_x  = find_evx( sys, i );
    // double ev_y  = find_evy( sys, j );
    // double ev_xy = ev_x * ev_y;

    // double ans  = find_ap( sys, l ) * e_xy + find_bp( sys, l ) * ev_x + find_cp( sys, l ) * ev_y + find_dp( sys, i );
    cuda::std::complex<double> ans = find_dp( sys, l );
    return ( ans );
}

__device__ cuda::std::complex<double> find_ev3DM( const System sys, const int i, const int j, const int l ) {
    // double ev_x  = find_evx( sys, i );
    // double ev_y  = find_evy( sys, j );
    // double ev_xy = ev_x * ev_y;

    // double ans  = find_am( sys, l ) * e_xy + find_bm( sys, l ) * ev_x + find_cm( sys, l ) * ev_y + find_dm( sys, i );
    cuda::std::complex<double> ans = find_dm( sys, l );
    return ( ans );
}

#ifdef USE_INDEX
__global__ void __launch_bounds__( 256 ) create_2th_setup( System sys ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    const int Nx  = sys.lat.Nx;
    const int Ny  = sys.lat.Ny;
    // const int Nz  = sys.lat.Nz;

    cuda::std::complex<double> temp;
    cuda::std::complex<double> temp1;
    cuda::std::complex<double> temp2;

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            int idx = tidY * Nx + tidX;
            temp        = 0.0;
            sys.L[idx]  = CMPLX( temp.real( ), temp.imag( ) );
            temp1       = 1.0 / find_ev3D( sys, tidX, tidY, 0 );
            sys.U[idx]  = CMPLX( temp1.real( ), temp1.imag( ) );
            temp2       = find_ev3DP( sys, tidX, tidY, 0 );
            sys.Up[idx] = CMPLX( temp2.real( ), temp2.imag( ) );
        }
    }
}

__global__ void __launch_bounds__( 256 )
    create_2th_order( System sys, cuDoubleComplex *sysU, cuDoubleComplex *sysL, cuDoubleComplex *sysUp ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    const int Nx  = sys.lat.Nx;
    const int Ny  = sys.lat.Ny;
    const int Nz  = sys.lat.Nz;
    const int Nxy = Nx * Ny;

    cuda::std::complex<double> temp;

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            for ( int l = 1; l < Nz; l++ ) {
                int idx = Nxy * l + tidY * Nx + tidX;

                temp    = cuda::std::complex<double>( sysU[Nxy * (l - 1) + tidY * Nx + tidX].x,
                                                   sysU[Nxy * (l - 1) + tidY * Nx + tidX].y );
                temp *= find_ev3DM( sys, tidX, tidY, l );
                sys.L[idx] = CMPLX( temp.real( ), temp.imag( ) );
                // printf("L - %d %d %d: %f %f\n", tidX, tidY, l, temp.real( ), temp.imag( ) );

                temp = cuda::std::complex<double>( sysUp[Nxy * (l - 1) + tidY * Nx + tidX].x,
                                                    sysUp[Nxy * (l - 1) + tidY * Nx + tidX].y );
                temp *= cuda::std::complex<double>( sysL[Nxy * l + tidY * Nx + tidX].x,
                                                    sysL[Nxy * l + tidY * Nx + tidX].y );

                temp       = 1.0 / ( find_ev3D( sys, tidX, tidY, l ) - temp );
                sys.U[idx] = CMPLX( temp.real( ), temp.imag( ) );
                // printf("U - %d %d %d: %f %f\n", tidX, tidY, l, temp.real( ), temp.imag( ) );

                temp        = find_ev3DP( sys, tidX, tidY, l );
                sys.Up[idx] = CMPLX( temp.real( ), temp.imag( ) );
                // printf("Up - %d %d %d: %f %f\n", tidX, tidY, l, temp.real( ), temp.imag( ) );
            }
        }
    }
}
#else
__global__ void __launch_bounds__( 256 ) create_2th_setup( System sys ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    const int Nx  = sys.lat.Nx;
    const int Ny  = sys.lat.Ny;
    const int Nz  = sys.lat.Nz;
    const int Nxz = Nx * sys.lat.Nz;

    cuda::std::complex<double> temp;
    cuda::std::complex<double> temp1;
    cuda::std::complex<double> temp2;

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            int idx     = tidY * Nxz + tidX * Nz;
            temp        = 0.0;
            sys.L[idx]  = CMPLX( temp.real( ), temp.imag( ) );
            temp1       = 1.0 / find_ev3D( sys, tidX, tidY, 0 );
            sys.U[idx]  = CMPLX( temp1.real( ), temp1.imag( ) );
            temp2       = find_ev3DP( sys, tidX, tidY, 0 );
            sys.Up[idx] = CMPLX( temp2.real( ), temp2.imag( ) );
        }
    }
}

__global__ void __launch_bounds__( 256 )
    create_2th_order( System sys, cuDoubleComplex *sysU, cuDoubleComplex *sysL, cuDoubleComplex *sysUp ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int strideX { static_cast<int>( blockDim.x * gridDim.x ) };

    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
    const int strideY { static_cast<int>( blockDim.y * gridDim.y ) };

    const int Nx  = sys.lat.Nx;
    const int Ny  = sys.lat.Ny;
    const int Nz  = sys.lat.Nz;
    const int Nxz = Nx * Nz;

    cuda::std::complex<double> temp;

    for ( int tidY = ty; tidY < Ny; tidY += strideY ) {
        for ( int tidX = tx; tidX < Nx; tidX += strideX ) {
            for ( int l = 1; l < Nz; l++ ) {
                int idx = tidY * Nxz + tidX * Nz + l;
                temp    = cuda::std::complex<double>( sysU[Nxz * tidY + Nz * tidX + (l - 1)].x,
                                                   sysU[Nxz * tidY + Nz * tidX + (l - 1)].y );

                temp *= find_ev3DM( sys, tidX, tidY, l );
                sys.L[idx] = CMPLX( temp.real( ), temp.imag( ) );
                // printf("L - %d %d %d: %f %f\n", tidX, tidY, l, temp.real( ), temp.imag( ) );

                temp = cuda::std::complex<double>( sysUp[Nxz * tidY + Nz * tidX + (l - 1)].x,
                                                   sysUp[Nxz * tidY + Nz * tidX + (l - 1)].y );
                temp *= cuda::std::complex<double>( sysL[Nxz * tidY + Nz * tidX + l].x,
                                                    sysL[Nxz * tidY + Nz * tidX + l].y );

                temp       = 1.0 / ( find_ev3D( sys, tidX, tidY, l ) - temp );
                sys.U[idx] = CMPLX( temp.real( ), temp.imag( ) );
                // printf("U - %d %d %d: %f %f\n", tidX, tidY, l, temp.real( ), temp.imag( ) );

                temp        = find_ev3DP( sys, tidX, tidY, l );
                sys.Up[idx] = CMPLX( temp.real( ), temp.imag( ) );
                // printf("Up - %d %d %d: %f %f\n", tidX, tidY, l, temp.real( ), temp.imag( ) );
            }
        }
    }
}
#endif

void create_2th_order_wrapper( System sys ) {

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    dim3 threadPerBlock { 16, 16 };
    dim3 blocksPerGrid( numSMs, numSMs );

    void *args[] { &sys };
    void *args2[] { &sys, &sys.U, &sys.L, &sys.Up };

    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &create_2th_setup ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );
    CUDA_RT_CALL( cudaLaunchKernel( ( void * )( &create_2th_order ), blocksPerGrid, threadPerBlock, args2, 0, NULL ) );

    CUDA_RT_CALL( cudaPeekAtLastError( ) );
}
