#pragma once

#include "../headers/structs.h"

#ifdef __cplusplus
extern "C"
{
#endif

void load_wrapper(System sys, DSTN dst, double _Complex *rhs, double *in, double *in2);

#ifdef __cplusplus
}
#endif