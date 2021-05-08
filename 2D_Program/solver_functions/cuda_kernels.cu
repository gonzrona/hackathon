#include <stdio.h>
#include <complex.h>

#include "cuda_helper.h"
#include "cuda_kernels.h"

__global__ void load_1st_DST( std::complex<double> *rhs, double *in, double *in2)
{
  printf("%f : %f\n", rhs[threadIdx.x].real(), rhs[threadIdx.x].imag() );
}

void load_1st_DST_wrapper(System sys, DSTN dst, double _Complex *rhs, double *in, double *in2) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2*Nx + 2, NC = (N/2) + 1;

    std::complex<double> *d_rhs;
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_rhs), sys.lat.Nxy * sizeof(double _Complex)));
    CUDA_RT_CALL(cudaMemcpy(d_rhs, rhs, sys.lat.Nxy * sizeof(double _Complex), cudaMemcpyHostToDevice));

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    int threadPerBlock { 10 };
    int blocksPerGrid { 1 };

    // std::complex<double> * asdf;
    // asdf = reinterpret_cast<std::complex<double>(&)[2]>(rhs);

    void *args[] { &d_rhs, &in, &in2};

    CUDA_RT_CALL( cudaLaunchKernel(
        (void *)( &load_1st_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );
}