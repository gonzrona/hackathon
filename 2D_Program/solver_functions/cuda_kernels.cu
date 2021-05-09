#include <stdio.h>
// #include <complex.h>

#include "cuda_helper.h"
#include "cuda_kernels.h"

// #pragma omp for
//     for(j = 0; j < Ny; j++) {
//         my = j*Nx;

//         for (i=0; i<dst.Nx; i++) { in[(j*N) + i+1] = rhs[i + my].x; }
//         for (i=0; i<dst.Nx; i++) { in2[(j*N) + i+1] = rhs[i + my].y; }
//     }

__global__ void load_1st_DST(const int N, const int Nx, const int Ny,
                             const cuDoubleComplex *__restrict__ rhs,
                             double *__restrict__ in,
                             double *__restrict__ in2) {
  const int tx{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  const int strideX{static_cast<int>(blockDim.x * gridDim.x)};

  const int ty{static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)};
  const int strideY{static_cast<int>(blockDim.y * gridDim.y)};

  
    for (int tidY = ty; tidY < Ny; tidY += strideY) {
      for (int tidX = tx; tidX < Nx; tidX += strideX) {

      in[tidY * N + tidX + 1] = rhs[Ny * tidY + tidX].x;
      in2[tidY * N + tidX + 1] = rhs[Ny * tidY + tidX].y;
    }
  }
}

void load_1st_DST_wrapper(System sys, DSTN dst, cuDoubleComplex *d_rhs,
                          double *in, double *in2) {

  int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
  int N = 2 * Nx + 2;//, NC = (N/2) + 1;

  int numSMs;
  CUDA_RT_CALL(
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

  dim3 threadPerBlock{1, 1};
  dim3 blocksPerGrid{1, 1};

  void *args[]{&N, &Nx, &Ny, &d_rhs, &in, &in2};

  CUDA_RT_CALL(cudaLaunchKernel((void *)(&load_1st_DST), blocksPerGrid,
                                threadPerBlock, args, 0, NULL));

          
  CUDA_RT_CALL(cudaPeekAtLastError());
  CUDA_RT_CALL(cudaDeviceSynchronize());
}