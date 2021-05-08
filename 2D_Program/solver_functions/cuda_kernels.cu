#include <stdio.h>

#include "cuda_helper.h"
#include "cuda_kernels.h"

__global__ void load_1st_DST(double *in, double *in2)
{
  printf("Hello\n");
}

void load_1st_DST_wrapper(System sys, DSTN dst, double *in, double *in2) {

    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int N = 2*Nx + 2, NC = (N/2) + 1;

    int numSMs;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );

    int threadPerBlock { 1 };
    int blocksPerGrid { numSMs * 1 };

    void *args[] {&in, &in2};

    CUDA_RT_CALL( cudaLaunchKernel(
        (void *)( &load_1st_DST ), blocksPerGrid, threadPerBlock, args, 0, NULL ) );
}