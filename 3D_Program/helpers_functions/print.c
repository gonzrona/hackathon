#include "../headers/structs.h"

void printLattice(Lattice lat) {
    printf("\n    LATTICE:\n");
    printf("\tNx: %d, Ny: %d\n", lat.Nx, lat.Ny);
    printf("\thx: %f, hy: %f\n", lat.hx, lat.hy);
    printf("\tx0: %f, x1: %f\n", lat.x0, lat.x1);
    printf("\ty0: %f, y1: %f\n\n", lat.y0, lat.y1);
}

void printDoubleArray(int n, double *a, char c[]) {
    int i;
    printf("\n");
    for( i = 0; i < n; i++) {
        printf("\t%s[%d] = %f\n", c, i, a[i]);
    }
    printf("\n");
}

void printOrder(System sys){
    if (sys.order == sixth) {
        printf("\n\t6th Order\n");
    }
    else if (sys.order == fourth) {
        printf("\n\t4th Order\n");
    }
    else {
        printf("\n\t2nd Order\n");
    }
}
