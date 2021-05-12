#include "../headers/structs.h"

double normInf(int n, double complex *x) {
    int i;
    double xm, norm = 0;
    
    for (i=0; i<n; i++){
        xm = cabs( x[i] );
        if (xm > norm) norm = xm;
    }
    
    return norm;
}

double normL2(int n, double complex *x) {
    int i;
    double norm;
    norm=0;
    for( i = 0; i < n; i++) {
        norm += cabs(x[i])*cabs(x[i]) ;
    }
    norm = sqrt(norm);
    return norm;
}
