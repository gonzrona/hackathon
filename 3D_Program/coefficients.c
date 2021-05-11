#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

/*
 Define the 27 coefficients in 3D stencil points.
 ========================================================================
 general important input:
 order    - the order of approximation of your scheme: 2, 4 or 6.
 Nz - the number of grid points.
 hx,hy,hz - the grid steps.
 k2_bg_ext - the value of k^2 , extended version is required for the 4th and 6th order scheme.
 A,B,C,z0,z1 - required parameter for setting up boundary condition (only used in 6th order).
 
 output: the value of 27 coefficients in 3D stencil points.
 a - coefficient of the corners on the l-level (current level)
 b - coefficient of the points differing in y on the l-level
 c - coefficient of the points differing in x on the l-level
 d - coefficient of the center points differing on the l-level
 
 ap - coefficient of the corners on the (l+1)-level (top level)
 bp - coefficient of the points differing in y on the (l+1)-level
 cp - coefficient of the points differing in x on the (l+1)-level
 dp - coefficient of the center points differing on the (l+1)-level
 
 am - coefficient of the corners on the (l-1)-level (bottom level)
 bm - coefficient of the points differing in y on the (l-1)-level
 cm - coefficient of the points differing in x on the (l-1)-level
 dm - coefficient of the center points differing on the (l-1)-level
 
 ========================================================================
 
 NOTE:
 No user input required here however depending on the function k used, some parameter may be removed.
 
 Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
 */

void derivative_kz_4th_order(int Nz, double hz, double complex *k, double complex *kz,double A,double B,double C, double z0, double z1);
void derivative_kzz_4th_order(int Nz, double hz, double complex *k, double complex *kzz,double A,double B,double C, double z0, double z1);

void coeff_2nd(double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, double complex *k2_bg_ext, double hx, double hy, double hz, int Nz){
    int i;
    double Rzx, Rzy;
    Rzx = hz*hz/hx/hx; Rzy = hz*hz/hy/hy;
    
    for (i=0; i<Nz; i++) {
        a[i] = 0;
        b[i] = Rzx;
        c[i] = Rzy;
        d[i] = -2.0*(Rzx + Rzy + 1.0) + hz*hz*k2_bg_ext[i+1];
        
        ap[i] = 0;
        bp[i] = 0;
        cp[i] = 0;
        dp[i] = 1;
        
        am[i] = 0;
        bm[i] = 0;
        cm[i] = 0;
        dm[i] = 1;
    }
    
    return;
}

void coeff_4th(double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, double complex *k2_bg_ext, double hx, double hy, double hz, int Nz){
    int i;
    
    double Rzx, Rzy;
    double one_sixth = 1.0/6.0;
    double one_twelfth = 1.0/12.0;
    Rzx = hz*hz/hx/hx; Rzy = hz*hz/hy/hy;
    
    for (i=0; i<Nz; i++) {
        a[i] = one_twelfth*(Rzx+Rzy);
        b[i] = one_sixth*(4.0*Rzx - Rzy - 1.0 + 0.5*hz*hz*k2_bg_ext[i+1]);
        c[i] = one_sixth*(4.0*Rzy - Rzx - 1.0 + 0.5*hz*hz*k2_bg_ext[i+1]);
        d[i] = - 4.0*(1.0 + Rzx + Rzy)/3.0 + 0.5*hz*hz*k2_bg_ext[i+1];
        
        ap[i] = 0.0;
        bp[i] = one_twelfth*(1.0 + Rzx);
        cp[i] = one_twelfth*(1.0 + Rzy);
        dp[i] = 1.0 - one_sixth*(2.0 + Rzx + Rzy) + one_twelfth*hz*hz*k2_bg_ext[i+2];

        am[i] = 0.0;
        bm[i] = one_twelfth*(1.0 + Rzx);
        cm[i] = one_twelfth*(1.0 + Rzy);
        dm[i] = 1.0 - one_sixth*(2.0 + Rzx + Rzy) + one_twelfth*hz*hz*k2_bg_ext[i];
    }
    return;
}

void coeff_6th(double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, double complex *k2_bg_ext, double hx, double hy, double hz, int Nz, double A,double B,double C, double z0, double z1){
    
    int i;
    double complex *k_prime = malloc(Nz * sizeof(double complex));
    double complex *k_double_prime = malloc(Nz * sizeof(double complex));
    
    derivative_kz_4th_order(Nz, hz, k2_bg_ext, k_prime,A,B,C,z0,z1);
    derivative_kzz_4th_order(Nz, hz, k2_bg_ext, k_double_prime,A,B,C,z0,z1);
    
    for (i=0; i<Nz; i++) {
        a[i] = 1.0/10.0+hz*hz*k2_bg_ext[i+1]/90.0;
        b[i] = 7.0/15.0-hz*hz*k2_bg_ext[i+1]/90.0;
        c[i] = 7.0/15.0-hz*hz*k2_bg_ext[i+1]/90.0;
        d[i] = -64.0/15.0 + 14.0*hz*hz*k2_bg_ext[i+1]/15.0 - hz*hz*hz*hz*k2_bg_ext[i+1]*k2_bg_ext[i+1]/20.0 + hz*hz*hz*hz*k_double_prime[i]/20.0;
        
        ap[i] = 1.0/30.0;
        bp[i] = 1.0/10.0+hz*hz*k2_bg_ext[i+2]/90.0 + (k_prime[i]*hz*hz*hz/120.0);
        cp[i] = 1.0/10.0+hz*hz*k2_bg_ext[i+2]/90.0 + (k_prime[i]*hz*hz*hz/120.0);
        dp[i] = 7.0/15.0-hz*hz*k2_bg_ext[i+2]/90.0 + (k_prime[i]*hz*hz*hz/20.0)*(1.0/3.0 + k2_bg_ext[i+2]*hz*hz/6.0);
        
        am[i] = 1.0/30.0;
        bm[i] = 1.0/10.0+hz*hz*k2_bg_ext[i]/90.0 - (k_prime[i]*hz*hz*hz/120.0);
        cm[i] = 1.0/10.0+hz*hz*k2_bg_ext[i]/90.0 - (k_prime[i]*hz*hz*hz/120.0);
        dm[i] = 7.0/15.0-hz*hz*k2_bg_ext[i]/90.0 - (k_prime[i]*hz*hz*hz/20.0)*(1.0/3.0 + k2_bg_ext[i]*hz*hz/6.0);
    }
    
    free(k_prime);k_prime=NULL;
    free(k_double_prime);k_double_prime=NULL;
    return;
}


void coefficients(double complex *a, double complex *b, double complex *c, double complex *d, double complex *ap, double complex *bp, double complex *cp, double complex *dp, double complex *am, double complex *bm, double complex *cm, double complex *dm, double complex *k2_bg_ext, double hx, double hy, double hz, int Nz, int order, double A,double B,double C, double z0, double z1){
    
    switch (order) {
        case 2:
            coeff_2nd(a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm, k2_bg_ext, hx, hy, hz, Nz);
            break;
        case 4:
            coeff_4th(a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm, k2_bg_ext, hx, hy, hz, Nz);
            break;
        case 6:
            coeff_6th(a, b, c, d, ap, bp, cp, dp, am, bm, cm, dm, k2_bg_ext, hx, hy, hz, Nz,A,B,C,z0,z1);
            break;
        default:
            break;
    }
    
    return;
}


