#pragma once

#include "../headers/structs.h"

#ifdef __cplusplus
extern "C" {
#endif

void load_1st_DST_wrapper(const System sys, const DSTN dst,
                          const cuDoubleComplex *d_rhs, double *in,
                          double *in2);
void store_1st_DST_wrapper(const System sys, const DSTN dst,
                           cuDoubleComplex *d_rhat, const cuDoubleComplex *out,
                           const cuDoubleComplex *out2);
void load_2st_DST_wrapper(const System sys, const DSTN dst,
                          const cuDoubleComplex *d_xhat, double *in,
                          double *in2);
void store_2st_DST_wrapper(const System sys, const DSTN dst,
                           cuDoubleComplex *d_sol, const cuDoubleComplex *out,
                           const cuDoubleComplex *out2);

#ifdef __cplusplus
}
#endif