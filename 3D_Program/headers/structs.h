#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>
#include <sys/time.h>


enum Order{second, fourth, sixth};

typedef struct {
    int     Nx, Ny, Nz, Nxyz;
    double  hx, hy, hz, x0, x1, y0, y1, z0, z1;
} Lattice;

typedef struct {
    Lattice lat;

    enum Order order;
    
    double complex *a, *b, *c, *d, *ap, *bp, *cp, *dp, *am, *bm, *cm, *dm;
    double complex *k_bg_ext, *k2_bg_ext;
    double A,B,C,beta,gamma;
    double complex *sol_analytic, *rhs;
    double complex *L, *U, *Up;
    double complex *sol, *res, *error;
} System;


typedef struct {
    clock_t time0;
    double start_t, start_t_n, computed_time, computed_t, computed_t_n;
} Time;
