#pragma once

#include "../headers/structs.h"

#ifdef __cplusplus
extern "C" {
#endif

void load_1st_DST_wrapper(System sys, DSTN dst, cuDoubleComplex *d_rhs,
                          double *in, double *in2);
void store_1st_DST_wrapper(System sys, DSTN dst, cuDoubleComplex *d_rhat,
                           cuDoubleComplex *out, cuDoubleComplex *out2);

#ifdef __cplusplus
}
#endif