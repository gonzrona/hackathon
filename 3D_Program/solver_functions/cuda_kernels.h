#pragma once

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "../headers/structs.h"

void load_1st_DST_wrapper( const cudaStream_t streams,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                NR,
                           cuDoubleComplex *  d_rhs,
                           double *           in );

void load_2st_DST_wrapper( const cudaStream_t streams,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                NR,
                           int                NC,
                           cuDoubleComplex *  d_rhs,
                           double *           in );

void store_1st_DST_wrapper( const cudaStream_t stream,
                            int                l,
                            int                Nx,
                            int                Ny,
                            int                NR,
                            int                NC,
                            cuDoubleComplex *  out,
                            cuDoubleComplex *  d_rhat );

void load_3st_DST_wrapper( const cudaStream_t streams,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                Nz,
                           int                NR,
                           cuDoubleComplex *  d_rhs,
                           double *           in );

void load_4st_DST_wrapper( const cudaStream_t streams,
                           int                l,
                           int                Nx,
                           int                Ny,
                           int                NR,
                           int                NC,
                           cuDoubleComplex *  d_rhs,
                           double *           in );

void store_2st_DST_wrapper( const cudaStream_t stream,
                            int                l,
                            int                Nx,
                            int                Ny,
                            int                NR,
                            int                NC,
                            cuDoubleComplex *  out,
                            cuDoubleComplex *  d_rhat );

void triangular_solver_wrapper( const cudaStream_t     stream,
                                System                 sys,
                                int                    Nx,
                                int                    Ny,
                                int                    NZ,
                                const cuDoubleComplex *d_rhat,
                                cuDoubleComplex *      d_xhat,
                                cuDoubleComplex *      d_y );

// void middle_stuff_ls_DST_wrapper( const cudaStream_t streams,
//                                   System             sys,
//                                   //   const DSTN             dst,
//                                   const cuDoubleComplex *out,
//                                   double *               in,
//                                   cuDoubleComplex *      d_y );

#ifdef __cplusplus
}
#endif
