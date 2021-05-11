#include <complex.h>
#include <math.h>

/*
 Define the k_bg^2 distribution in the background media.
 ========================================================================
 general important input:
 Nz    - the number of grid points in z-directions.
 hz    - the grid steps in z-directions,
 z0    - the z-coordinates of the bottom of the
 computational domain.
 A, B, C - the parameter used in constructing k of the form A-B*sin(C*z).
 
 output:
 k_bg_ext - k_bg_ext distribution in the background media.
 k2_bg_ext - k_bg_ext^2 distribution in the background media.
 
 ========================================================================
 
 NOTE:
 An extended version of k is needed for the 4th and 6th order scheme
 User may change the form of k(z) as they desired however this would also required the user to change other settings in other codes/files
 which will be listed below:
 1) 4th_order_partials.c
 2) second_test3D.c
 
 Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
 */


void background3D(int Nz, double hz, double z0, double complex *k_bg_ext,double complex *k2_bg_ext, double A,double B,double C){
    
    int l;
    double z;
    for( l = 0; l < Nz+2; l++){
        z = z0 + hz*l ;
        k_bg_ext[l] = (A-B*sin(C*z));
        k2_bg_ext[l] = k_bg_ext[l]*k_bg_ext[l];
    }
    
    return;
}
