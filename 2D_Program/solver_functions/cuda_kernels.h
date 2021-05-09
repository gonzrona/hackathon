#pragma once

#include "../headers/structs.h"

#ifdef __cplusplus
extern "C"
{
#endif

void load_1st_DST_wrapper(System sys, DSTN dst, cuDoubleComplex *rhs, double *in, double *in2);

#ifdef __cplusplus
}
#endif