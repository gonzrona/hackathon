#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include <complex.h>
#include <fftw3.h>

#define ind(i,j,Nx) i+(j)*(Nx)
#define ind_e(i,j,Nx) i+1+(j+1)*(Nx+2)

enum Order{second, fourth, sixth};
enum Direction{forward, reverse};

typedef struct {
    int Nx, Ny, N, NC;
    double coef;
    
    double *in1, *in2;
    fftw_complex *out1, *out2;
    fftw_plan plan1, plan2;
    
} DSTN;

typedef struct {
    int     Nx, Ny, Nxy;
    double  hx, hy, x0, x1, y0, y1;
} Lattice;

typedef struct {
    Lattice lat;
    int threadCount;
    enum Order order;
    double _Complex *k, *rhs, *L, *U, *Up, *sol, *sol_analytic, *res, *error;
    double _Complex *bp, *ap, *b, *a, *bm, *am;
} System;

typedef struct {
    clock_t time0;
    double start_t, start_t_n, computed_time, computed_t, computed_t_n;
} Time;
