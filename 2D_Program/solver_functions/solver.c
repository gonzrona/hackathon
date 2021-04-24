#include "../headers/structs.h"

void DST(DSTN dst, double _Complex *b, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out);

void solver(System sys) {
    
    DSTN dst;
    int i,j,my,mx;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double _Complex *rhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    double _Complex *xhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    
    int N = 2*Nx + 2, NC = (N/2) + 1;
    dst.Nx = Nx; dst.N = N; dst.coef = sqrt(2.0/(Nx+1));
    
#pragma omp parallel private (i,j,mx,my)
    {
            
        double *in        = (double *) fftw_malloc(sizeof(double) * N);
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NC);

        double _Complex *b    = (double _Complex *) malloc(Nx * sizeof(double _Complex));
        double _Complex *bhat = (double _Complex *) malloc(Nx * sizeof(double _Complex));
        double _Complex *y    = (double _Complex *) malloc(Ny * sizeof(double _Complex));
        fftw_plan plan;
            
    #pragma omp critical (make_plan)
        { plan = fftw_plan_dft_r2c_1d ( N, in, out, FFTW_ESTIMATE ); }

    #pragma omp for
        for(j = 0; j < Ny; j++) {
            my = j*Nx ;
            for(i = 0; i < Nx; i++){
                b[i] = sys.rhs[i + my] ;
            }
            DST(dst, b, bhat, plan, in, out);
            for(i = 0; i < Nx; i++){
                rhat[i + my] = bhat[i] ;
            }
        }
        
    #pragma omp for
        for(i = 0; i < Nx; i++){
            y[0] = rhat[i];
            mx = i*Ny ;
            for(j = 1; j < Ny; j++) {
                y[j] = rhat[ind(i,j,Nx)] - sys.L[j + mx]*y[j - 1];
            }
            xhat[Ny - 1 + mx] = y[Ny - 1]/sys.U[Ny - 1 + mx] ;
            for(j = Ny-2; j >= 0; j--) {
                xhat[j + mx] =  ( y[j] - sys.Up[j + mx] * xhat[j + 1 + mx] )/sys.U[j + mx] ;
            }
        }
      
    #pragma omp for
        for(j = 0; j < Ny; j++) {
            my = j*Nx;
            for(i = 0; i < Nx; i++){
                b[i] = xhat[j + i*Ny] ;
            }
            DST(dst, b, bhat, plan, in, out);
            for(i = 0; i < Nx; i++){
                sys.sol[i + my] = bhat[i] ;
            }
        }
            
        fftw_destroy_plan(plan);
        free(in); in = NULL;
        fftw_free(out); out = NULL;
        free(b); b = NULL;
        free(bhat); bhat = NULL;
        free(y); y = NULL;
    }

    free(rhat); rhat = NULL;
    free(xhat); xhat = NULL;
}
