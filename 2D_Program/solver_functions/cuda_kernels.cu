#include <stdio.h>

#include "cuda_kernels.h"


__global__ void print_something()
{
  printf("Hello\n");
}

void print_wrapper() {
    print_something<<<1, 10>>>();
}