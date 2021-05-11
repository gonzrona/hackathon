#include <stdio.h>
#include <complex.h>
#include <math.h>

double normL2(int N, double complex *x){
    /*
     Find the L2 norm
     ========================================================================
     input:  N            - the length of the vector.
     x             - a complex vector with N components
     output: normL2     - L2 norm of the vector
     
     ========================================================================
     
     NOTE:
     No user input required here; No changes should be made to this code.
     
     Dr. Yury Gryazin, 08/26/2016, ISU, Pocatello, ID
     */
    
    
    int i;
    double norm;
    norm=0;
    for( i = 0; i < N; i++) {
        norm += cabs(x[i])*cabs(x[i]) ;
    }
    norm = sqrt(norm);
    
    return norm;
}

double L2err(int N, double complex *x, double complex *u){
    /*
     Find the L2 relative error
     ========================================================================
     input:  N     - the length of the vector.
     x             - a complex vector with N components
     u             - exact solution
     output: norm  - the relative error
     
     ========================================================================
     
     NOTE:
     No user input required here; No changes should be made to this code.
     
     Dr. Yury Gryazin, Ronald Gonzales, Yun Teck Lee 06/12/2018, ISU, Pocatello, ID
     */
    
    int i;
    double norm;
    norm=0;
    for( i = 0; i < N; i++) {
        norm += cabs(x[i]-u[i])*cabs(x[i]-u[i]) ;
    }
    norm = sqrt(norm)/normL2(N,u);
    
    return norm;
}

