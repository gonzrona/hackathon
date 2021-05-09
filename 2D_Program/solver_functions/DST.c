#include "../headers/structs.h"

#include "cuda_helper.h"
#include "cuda_kernels.h"
#include <cuComplex.h>
#include <cufftw.h>

#define USE_BATCHED 1
#define USE_CUFFTW 1

void forwardDST(System sys, DSTN dst, double _Complex *rhs,
                double _Complex *rhat, fftw_plan plan, double *in,
                fftw_complex *out, fftw_plan plan2, double *in2,
                fftw_complex *out2) {

  int i, j, my;
  int Nx = sys.lat.Nx, Ny = sys.lat.Ny;

#if USE_BATCHED

  int N = 2 * Nx + 2, NC = (N / 2) + 1;

  // size_t size_in = sizeof(double) * N * Ny;
  // size_t size_out = sizeof(fftw_complex) * NC * Ny;

  cuDoubleComplex *d_rhs;
  CUDA_RT_CALL(
      cudaMalloc((void **)(&d_rhs), sys.lat.Nxy * sizeof(double _Complex)));
  CUDA_RT_CALL(cudaMemcpy(d_rhs, rhs, sys.lat.Nxy * sizeof(double _Complex),
                          cudaMemcpyHostToDevice));

  load_1st_DST_wrapper(sys, dst, d_rhs, in, in2);
  CUDA_RT_CALL(cudaDeviceSynchronize());

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     my = j * Nx;

//     for (i = 0; i < dst.Nx; i++) {
//       in[(j * N) + i + 1] = creal(rhs[i + my]);
//       in2[(j * N) + i + 1] = cimag(rhs[i + my]);
//     }
//   }
  //   CUDA_RT_CALL(cudaMemPrefetchAsync(in, size_in, cudaCpuDeviceId, NULL));
  // CUDA_RT_CALL(cudaMemPrefetchAsync(in2, size_in, cudaCpuDeviceId, NULL));

  // for (j = 0; j < Ny; j++) {
  //     for (i = 0; i < dst.Nx; i++) {
  //   my = j * Nx;
  //     printf("%d %d :: %f %f\n", i, j, in[(j * N) + i + 1] , in2[(j * N) + i + 1] );
  //   }
  // }


#if USE_OMP
#pragma omp critical(fftw_execute)
#endif
  {
    fftw_execute(plan);  /********************* FFTW *********************/
    fftw_execute(plan2); /********************* FFTW *********************/
  }


#pragma omp for
  for (j = 0; j < Ny; j++) {
    my = j * Nx;

    for (i = 0; i < dst.Nx; i++) {
      rhat[i + my] = dst.coef * (-cimag(out[(j * NC) + i + 1]) -
                                 I * cimag(out2[(j * NC) + i + 1]));
    }
  }

#else

#pragma omp for
  for (j = 0; j < Ny; j++) {
    my = j * Nx;

    for (i = 0; i < dst.Nx; i++) {
      in[i + 1] = creal(rhs[i + my]);
    }
    for (i = 0; i < dst.Nx; i++) {
      in2[i + 1] = cimag(rhs[i + my]);
    }

    fftw_execute(plan);  /********************* FFTW *********************/
    fftw_execute(plan2); /********************* FFTW *********************/

    for (i = 0; i < dst.Nx; i++) {
      rhat[i + my] = dst.coef * (-cimag(out[i + 1]) - I * cimag(out2[i + 1]));
    }
  }

#endif
}

// void forwardDST_CUDA(System sys, DSTN dst, double _Complex * sys.rhs, cuDoubleComplex *rhs,
//                      double _Complex *rhat, fftw_plan plan, double *in,
//                      fftw_complex *out, fftw_plan plan2, double *in2,
//                      fftw_complex *out2) {

//   int i, j, my;
//   int Nx = sys.lat.Nx, Ny = sys.lat.Ny;

// #if USE_BATCHED

//   int N = 2 * Nx + 2, NC = (N / 2) + 1;
//   printf("N= %d: Nx = %d: Ny = %d\n", N, Nx, Ny);

//   size_t size_in = sizeof(double) * N * Ny;
//   size_t size_out = sizeof(fftw_complex) * NC * Ny;

//   // CUDA_RT_CALL(cudaMemPrefetchAsync(in, size_in, cudaCpuDeviceId, NULL));
//   // CUDA_RT_CALL(cudaMemPrefetchAsync(in2, size_in, cudaCpuDeviceId, NULL));

//   CUDA_RT_CALL(cudaMemPrefetchAsync(in, size_in, 0, NULL));
//   CUDA_RT_CALL(cudaMemPrefetchAsync(in2, size_in, 0, NULL));
//   CUDA_RT_CALL(cudaMemPrefetchAsync(out, size_out, 0, NULL));
//   CUDA_RT_CALL(cudaMemPrefetchAsync(out2, size_out, 0, NULL));

// //   load_1st_DST_wrapper(sys, dst, rhs, in, in2);

//   #pragma omp for
//       for(j = 0; j < Ny; j++) {
//           my = j*Nx;

//           for (i=0; i<dst.Nx; i++) { in[(j*N) + i+1] = sys.rhs[i + my].x; }
//           for (i=0; i<dst.Nx; i++) { in2[(j*N) + i+1] = sys.rhs[i + my].y; }
//       }

//   // CUDA_RT_CALL(cudaMemPrefetchAsync(in, size_in, 0, NULL));
//   // CUDA_RT_CALL(cudaMemPrefetchAsync(in2, size_in, 0, NULL));
//   CUDA_RT_CALL(cudaDeviceSynchronize());

// #if USE_OMP
// #pragma omp critical(fftw_execute)
// #endif
//   {
//     fftw_execute(plan);  /********************* FFTW *********************/
//     fftw_execute(plan2); /********************* FFTW *********************/
//   }

//   CUDA_RT_CALL(cudaMemPrefetchAsync(out, size_out, cudaCpuDeviceId, NULL));
//   CUDA_RT_CALL(cudaMemPrefetchAsync(out2, size_out, cudaCpuDeviceId, NULL));

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     my = j * Nx;

//     for (i = 0; i < dst.Nx; i++) {
//       rhat[i + my] = dst.coef * (-cimag(out[(j * NC) + i + 1]) -
//                                  I * cimag(out2[(j * NC) + i + 1]));
//     }
//   }

// #else

// #pragma omp for
//   for (j = 0; j < Ny; j++) {
//     my = j * Nx;

//     for (i = 0; i < dst.Nx; i++) {
//       in[i + 1] = creal(rhs[i + my]);
//     }
//     for (i = 0; i < dst.Nx; i++) {
//       in2[i + 1] = cimag(rhs[i + my]);
//     }

//     fftw_execute(plan);  /********************* FFTW *********************/
//     fftw_execute(plan2); /********************* FFTW *********************/

//     for (i = 0; i < dst.Nx; i++) {
//       rhat[i + my] = dst.coef * (-cimag(out[i + 1]) - I * cimag(out2[i + 1]));
//     }
//   }

// #endif
// }

void reverseDST(System sys, DSTN dst, double _Complex *xhat,
                double _Complex *sol, fftw_plan plan, double *in,
                fftw_complex *out, fftw_plan plan2, double *in2,
                fftw_complex *out2) {

  int i, j, my;
  int Nx = sys.lat.Nx, Ny = sys.lat.Ny;

#if USE_BATCHED

  int N = 2 * Nx + 2, NC = (N / 2) + 1;

  // size_t size_in = sizeof(double) * N * Ny;
  // size_t size_out = sizeof(fftw_complex) * NC * Ny;

  // CUDA_RT_CALL(cudaMemPrefetchAsync(in, size_in, cudaCpuDeviceId, NULL));
  // CUDA_RT_CALL(cudaMemPrefetchAsync(in2, size_in, cudaCpuDeviceId, NULL));
  // CUDA_RT_CALL(cudaMemPrefetchAsync(out, size_out, 0, NULL));
  // CUDA_RT_CALL(cudaMemPrefetchAsync(out2, size_out, 0, NULL));

#pragma omp for
  for (j = 0; j < Ny; j++) {
    my = j * Nx;

    for (i = 0; i < dst.Nx; i++) {
      in[(j * N) + i + 1] = creal(xhat[j + i * Ny]);
    }
    for (i = 0; i < dst.Nx; i++) {
      in2[(j * N) + i + 1] = cimag(xhat[j + i * Ny]);
    }
  }

  // CUDA_RT_CALL(cudaMemPrefetchAsync(in, size_in, 0, NULL));
  // CUDA_RT_CALL(cudaMemPrefetchAsync(in2, size_in, 0, NULL));

#if USE_OMP
#pragma omp critical(fftw_execute)
#endif
  {
    fftw_execute(plan);  /********************* FFTW *********************/
    fftw_execute(plan2); /********************* FFTW *********************/
  }

  // CUDA_RT_CALL(cudaMemPrefetchAsync(out, size_out, cudaCpuDeviceId, NULL));
  // CUDA_RT_CALL(cudaMemPrefetchAsync(out2, size_out, cudaCpuDeviceId, NULL));

#pragma omp for
  for (j = 0; j < Ny; j++) {
    my = j * Nx;

    for (i = 0; i < dst.Nx; i++) {
      sol[i + my] = dst.coef * (-cimag(out[(j * NC) + i + 1]) -
                                I * cimag(out2[(j * NC) + i + 1]));
    }
  }

#else

#pragma omp for
  for (j = 0; j < Ny; j++) {
    my = j * Nx;

    for (i = 0; i < dst.Nx; i++) {
      in[i + 1] = creal(xhat[j + i * Ny]);
    }
    for (i = 0; i < dst.Nx; i++) {
      in2[i + 1] = cimag(xhat[j + i * Ny]);
    }

    fftw_execute(plan);  /********************* FFTW *********************/
    fftw_execute(plan2); /********************* FFTW *********************/

    for (i = 0; i < dst.Nx; i++) {
      sol[i + my] = dst.coef * (-cimag(out[i + 1]) - I * cimag(out2[i + 1]));
    }
  }

#endif
}
