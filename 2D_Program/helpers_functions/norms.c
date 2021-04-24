#include "../headers/structs.h"

double normInf(int n, double _Complex *x) {
    int i;
    double xm, norm = 0;
    
    for (i=0; i<n; i++){
        xm = cabs( x[i] );
        if (xm > norm) norm = xm;
    }
    
    return norm;
}
