#include <stdio.h>

#include "cuda_kernels.h"

__global__ void print_something()
{
  printf("Hello\n");
}

void load_wrapper(System sys, DSTN dst, double _Complex *rhs, double *in, double *in2) {
    print_something<<<1, 10>>>();
}