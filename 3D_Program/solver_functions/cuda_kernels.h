#pragma once

#include "../headers/structs.h"

#ifdef __cplusplus
extern "C" {
#endif

void load_1st_DST_wrapper( const cudaStream_t     streams,
                           const System           sys,
                        //    const DSTN             dst,
                           const cuDoubleComplex *d_rhs,
                           double *               in );
void store_1st_DST_wrapper( const cudaStream_t     streams,
                            const System           sys,
                            // const DSTN             dst,
                            const cuDoubleComplex *out,
                            cuDoubleComplex *      d_rhat );
void load_2st_DST_wrapper( const cudaStream_t     streams,
                           const System           sys,
                        //    const DSTN             dst,
                           const cuDoubleComplex *d_xhat,
                           double *               in );
void store_2st_DST_wrapper( const cudaStream_t     streams,
                            const System           sys,
                            // const DSTN             dst,
                            const cuDoubleComplex *out,
                            cuDoubleComplex *      d_sol );
void middle_stuff_DST_wrapper( const cudaStream_t     streams,
                               System                 sys,
                               const cuDoubleComplex *d_rhat,
                               cuDoubleComplex *      d_xhat,
                               cuDoubleComplex *      d_y );
void middle_stuff_ls_DST_wrapper( const cudaStream_t     streams,
                                  System                 sys,
                                //   const DSTN             dst,
                                  const cuDoubleComplex *out,
                                  double *               in,
                                  cuDoubleComplex *      d_y );

#ifdef __cplusplus
}
#endif
