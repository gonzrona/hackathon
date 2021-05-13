#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>
#include <sys/time.h>

#define ind(i,j,Nx) i+(j)*(Nx)
#define ind_e(i,j,Nx) i+1+(j+1)*(Nx+2)

enum Order{second, fourth, sixth};

typedef struct {
    int     Nx, Ny, Nz, Nxyz;
    double  hx, hy, hz, x0, x1, y0, y1, z0, z1;
} Lattice;

typedef struct {
    Lattice lat;

    enum Order order;
    
    double _Complex * a;
    double _Complex *b, *c, *d, *ap, *bp, *cp, *dp, *am, *bm, *cm, *dm;
    double _Complex *k_bg_ext, *k2_bg_ext;
    double A,B,C,beta,gamma;
    double _Complex *sol_analytic, *rhs;
    double _Complex *L, *U, *Up;
    double _Complex *sol, *res, *error;
} System;


typedef struct {
    clock_t time0;
    double start_t, start_t_n, computed_time, computed_t, computed_t_n;
} Time;
