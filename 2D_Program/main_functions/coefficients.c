#include "../headers/structs.h"
#include "../headers/prototypes.h"


//bp ap bp
//b  a  b
//bm am bm

void second_order(System sys) {
    int j, Ny = sys.lat.Ny;
    double y, hx = sys.lat.hx, hy = sys.lat.hy, hy2 = hy*hy, y0 = sys.lat.y0, Ryx = hy*hy/hx/hx;

#pragma omp parallel for private(y)
    for (j=0; j<Ny; j++) {
        y = y0 + (j+1)*hy;
        sys.bp[j] = 0.0;
        sys.ap[j] = 1.0;
        sys.b[j]  = Ryx;
        sys.a[j]  = -2.0*Ryx - 2.0 + k(y)*k(y)*hy2;
        sys.bm[j] = 0.0;
        sys.am[j] = 1.0;
    }
}

void fourth_order(System sys) {
    int j, Ny = sys.lat.Ny;
    double y, hx = sys.lat.hx,  hy = sys.lat.hy, y0 = sys.lat.y0;
    double hy2 = hy*hy, Ryx = hy*hy/hx/hx;
    double _Complex k2, k2p, k2m;
    
#pragma omp parallel for private(y,k2,k2p,k2m)
    for( j = 0; j < Ny; j++) {
        y = y0 + (j+1)*hy; k2 = k(y)*k(y); k2p = k(y+hy)*k(y+hy); k2m = k(y-hy)*k(y-hy);

        sys.bp[j] = Ryx/12.0 + 1.0/12.0;
        sys.ap[j] = 5.0/6.0  - Ryx/6.0 + hy2*k2p/12.0;
        sys.b[j]  = Ryx*(5.0/6.0) - 1.0/6.0 + hy2*k2/12.0;
        sys.a[j]  = -Ryx*(5.0/3.0) - 5.0/3.0 + 2.0*hy2*k2/3.0;
        sys.bm[j] = Ryx/12.0 + 1.0/12.0;
        sys.am[j] = 5.0/6.0  - Ryx/6.0 + hy2*k2m/12.0;
    }
}

void sixth_order(System sys) {
    int j, Ny = sys.lat.Ny;
    double y, y0 = sys.lat.y0, hy = sys.lat.hy, hy2 = hy*hy, hy3 = hy2*hy, hy4 = hy2*hy2;
    double _Complex k2, k2p, k2m, ky2;

#pragma omp parallel for private(y,k2,k2p,k2m,ky2)
    for( j = 0; j < Ny; j++) {
        y = y0 + (j+1)*hy;
        k2 = k(y)*k(y); k2p = k(y+hy)*k(y+hy); k2m = k(y-hy)*k(y-hy); ky2 = k2_y(y);
                
        sys.ap[j] = 2.0/3.0 + hy2*k2p/90.0 + hy3*ky2*(hy2*k2p/6.0 + 2.0/3.0)/20.0;
        sys.bp[j] = 1.0/6.0 + hy2*k2p/90.0 + hy3*ky2/120.0;
        sys.a[j]  = -10.0/3.0 + 41.0*hy2*k2/45.0 - hy4*k2*k2/20.0 + hy4*k2_yy(y)/20.0;
        sys.b[j]  = 2.0/3.0 + hy2*k2/90.0;
        sys.am[j] = 2.0/3.0 + hy2*k2m/90.0 - hy3*ky2*(hy2*k2m/6.0 + 2.0/3.0)/20.0;
        sys.bm[j] = 1.0/6.0 + hy2*k2m/90.0 - hy3*ky2/120.0;
    }
}

void coefficients(System sys) {
    if (sys.order == sixth) { sixth_order(sys); }
    else if (sys.order == fourth) { fourth_order(sys); }
    else { second_order(sys); }
}
