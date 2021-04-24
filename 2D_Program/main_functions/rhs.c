#include "../headers/structs.h"
#include "../headers/prototypes.h"

void second_order_rhs(System sys) {
    int i,j,my;
    double x,y;
    int Nx = sys.lat.Nx,  Ny = sys.lat.Ny;
    double hx = sys.lat.hx,  hy = sys.lat.hy, x0 = sys.lat.x0, y0 = sys.lat.y0;
    
#pragma omp parallel for private(i,my,x,y)
    for( j = 0; j < Ny; j++) {
        y = y0 + (j+1)*hy;
        my = j*Nx;
        for( i = 0; i < Nx; i++){
            x = x0 + (i+1)*hx;
            sys.rhs[i + my]  = hy*hy*f(x,y);
        }
    }
}

void fourth_order_rhs(System sys) {
    int i,j,my;
    double x,y;
    int Nx = sys.lat.Nx,  Ny = sys.lat.Ny;
    double hx = sys.lat.hx, hy = sys.lat.hy, x0 = sys.lat.x0, y0 = sys.lat.y0;
    double hy2 = hy*hy;
    double a = 2.0/3.0, b = 1.0/12.0;

#pragma omp parallel for private(i,my,x,y)
    for(j = 0; j < Ny; j++) {
        y = y0 + (j+1)*hy;
        my = j*Nx;
        for( i = 0; i < Nx; i++){
            x = x0 + (i+1)*hx;
            sys.rhs[i + my] = hy2 * ( a * f(x,y) + b * (f(x,y-hy) + f(x,y+hy) + f(x-hx,y) + f(x+hx,y)) );
        }
    }
}

void sixth_order_rhs(System sys) {
    int i,j,my;
    int Nx = sys.lat.Nx,  Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double x, y, hx = sys.lat.hx,  hy = sys.lat.hy, hy2 = hy*hy, hy4 = hy2*hy2, x0 = sys.lat.x0, y0 = sys.lat.y0;
    double _Complex k2c;
    
    double _Complex *f_e   = malloc((Nx+2)*(Ny+2) * sizeof(double _Complex));
    double _Complex *fxx_e = malloc((Nx+2)*(Ny+2) * sizeof(double _Complex));
    double _Complex *fyy_e = malloc((Nx+2)*(Ny+2) * sizeof(double _Complex));
    double _Complex *fxxxx = malloc(Nxy * sizeof(double _Complex));
    double _Complex *fxxyy = malloc(Nxy * sizeof(double _Complex));
    double _Complex *fyyyy = malloc(Nxy * sizeof(double _Complex));
    
    double _Complex *k2y = malloc(Ny  * sizeof(double _Complex));
    double _Complex *fy  = malloc(Nxy * sizeof(double _Complex));
    
    double _Complex *k2_e = malloc((Ny+2) * sizeof(double _Complex));
    
    for (j=0; j<Ny+2; j++) { k2_e[j] = k2(y0 + j*hy); }
    
    for (i = 0; i < Nx+2; i++) {
        x = x0 + i*hx;
        for (j = 0; j < Ny+2; j++) {
            y = y0 + j*hy;
            f_e[i + j*(Nx+2)]  = f(x,y);
        }
    }
    
    f_y_4th_order(sys, f_e, fy);
    k2_y_4th_order(sys, k2_e, k2y);
    
    f_yy_4th_order(sys, f_e, fyy_e);
    f_xx_4th_order(sys, f_e, fxx_e);

    derivative_xx_2nd_order(sys, fxx_e, fxxxx);
    derivative_yy_2nd_order(sys, fxx_e, fxxyy);
    derivative_yy_2nd_order(sys, fyy_e, fyyyy);
    
#pragma omp parallel for private(i,my,y,k2c)
    for( j = 0; j < Ny; j++) {
        y = y0 + (j+1)*hy;
        k2c = k(y)*k(y);
        my = j*Nx;
        for( i = 0; i < Nx; i++){
            sys.rhs[i + my]  = hy2*((1.0-k2c*hy2/20.0)*f_e[ind_e(i,j,Nx)] + hy2*(fxx_e[ind_e(i,j,Nx)] + fyy_e[ind_e(i,j,Nx)])/12.0 + hy4*(fxxxx[i + my] + fyyyy[i + my])/360.0 + hy4*fxxyy[i + my]/90.0 + hy4*k2y[j]*fy[i + my]/60.0 );
        }
    }

    free(f_e); f_e = NULL;
    free(fxx_e); fxx_e = NULL;
    free(fyy_e); fyy_e = NULL;
    free(fxxxx); fxxxx = NULL;
    free(fxxyy); fxxyy = NULL;
    free(fyyyy); fyyyy = NULL;
    free(k2y); k2y = NULL;
    free(fy); fy = NULL;
    free(k2_e); k2_e = NULL;
}


void rhs(System sys) {

    int i,j,my;
    double x,y;
    int Nx = sys.lat.Nx,  Ny = sys.lat.Ny;
    double hx = sys.lat.hx,  hy = sys.lat.hy, x0 = sys.lat.x0, y0 = sys.lat.y0;
    
    for(j=0; j<Ny; j++) {
        y = y0 + (j+1)*hy;
        sys.k[j] = k(y);
    }

#pragma omp parallel for private(i,my,x,y)
    for(j=0; j<Ny; j++) {
        y = y0 + (j+1)*hy;
        my = j*Nx;
        for(i=0; i<Nx; i++){
            x = x0 + (i+1)*hx;
            sys.sol_analytic[i + my] = analytic_solution(x,y);
        }
    }
    
    if (sys.order == sixth) { sixth_order_rhs(sys); }
    else if (sys.order == fourth) { fourth_order_rhs(sys); }
    else { second_order_rhs(sys); }
    
    sys.rhs[0] = sys.rhs[0] - sys.bm[0]*analytic_solution(sys.lat.x0,sys.lat.y0) - sys.am[0]*analytic_solution(sys.lat.x0+hx,sys.lat.y0) - sys.bm[0]*analytic_solution(sys.lat.x0+hx+hx,sys.lat.y0) - sys.b[0]*analytic_solution(sys.lat.x0,sys.lat.y0+hy) - sys.bp[0]*analytic_solution(sys.lat.x0,sys.lat.y0+hy+hy);
    
    sys.rhs[Nx-1] = sys.rhs[Nx-1] - sys.bm[0]*analytic_solution(sys.lat.x1-hx-hx,sys.lat.y0) - sys.am[0]*analytic_solution(sys.lat.x1-hx,sys.lat.y0) - sys.bm[0]*analytic_solution(sys.lat.x1,sys.lat.y0) - sys.b[0]*analytic_solution(sys.lat.x1,sys.lat.y0+hy) - sys.bp[0]*analytic_solution(sys.lat.x1,sys.lat.y0+hy+hy);
    
    sys.rhs[(Ny-1)*Nx] = sys.rhs[(Ny-1)*Nx] - sys.bm[Ny-1]*analytic_solution(sys.lat.x0,sys.lat.y1-hy-hy) - sys.b[Ny-1]*analytic_solution(sys.lat.x0,sys.lat.y1-hy) - sys.bp[Ny-1]*analytic_solution(sys.lat.x0,sys.lat.y1) - sys.ap[Ny-1]*analytic_solution(sys.lat.x0+hx,sys.lat.y1) - sys.bp[Ny-1]*analytic_solution(sys.lat.x0+hx+hx,sys.lat.y1);
    
    sys.rhs[Nx-1+(Ny-1)*Nx] = sys.rhs[Nx-1+(Ny-1)*Nx] - sys.bm[Ny-1]*analytic_solution(sys.lat.x1,sys.lat.y1-hy-hy) - sys.b[Ny-1]*analytic_solution(sys.lat.x1,sys.lat.y1-hy) - sys.bp[Ny-1]*analytic_solution(sys.lat.x1,sys.lat.y1) - sys.ap[Ny-1]*analytic_solution(sys.lat.x1-hx,sys.lat.y1) - sys.bp[Ny-1]*analytic_solution(sys.lat.x1-hx-hx,sys.lat.y1);
        
    for(j=1; j<Ny-1; j++) {
        y = y0 + (j+1)*hy;
        my = j*Nx;
        sys.rhs[my]  = sys.rhs[my] - sys.b[j]*analytic_solution(sys.lat.x0,y)  - sys.bm[j]*analytic_solution(sys.lat.x0,y-hy) - sys.bp[j]*analytic_solution(sys.lat.x0,y+hy);

        sys.rhs[my + Nx-1] = sys.rhs[my + Nx-1] - sys.b[j]*analytic_solution(sys.lat.x1,y) - sys.bm[j]*analytic_solution(sys.lat.x1,y-hy) - sys.bp[j]*analytic_solution(sys.lat.x1,y+hy);
    }

    my = (Ny-1)*Nx;
    for(i=1; i<Nx-1; i++) {
        x = x0 + (i+1)*hx;
        sys.rhs[i]  = sys.rhs[i] - sys.am[0]*analytic_solution(x,sys.lat.y0)  - sys.bm[0]*analytic_solution(x-hx,sys.lat.y0) - sys.bm[0]*analytic_solution(x+hx,sys.lat.y0);

        sys.rhs[i + my] = sys.rhs[i + my] - sys.ap[Ny-1]*analytic_solution(x,sys.lat.y1) - sys.bp[Ny-1]*analytic_solution(x-hx,sys.lat.y1) - sys.bp[Ny-1]*analytic_solution(x+hx,sys.lat.y1);
    }
    
}
