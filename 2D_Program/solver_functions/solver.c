#include "../headers/structs.h"

void forwardDST(System sys, DSTN dst, double _Complex *rhs, double _Complex *bhat, fftw_plan plan, double *in, fftw_complex *out);
void reverseDST(System sys, DSTN dst, double _Complex *xhat, double _Complex *sol, fftw_plan plan, double *in, fftw_complex *out);


void solver(System sys) {
    
    DSTN dst;
    int i,j,mx;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nxy = sys.lat.Nxy;
    double _Complex *rhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    double _Complex *xhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    
    int N = 2*Nx + 2, NC = (N/2) + 1;
    dst.Nx = Nx; dst.N = N; dst.coef = sqrt(2.0/(Nx+1));
    
#pragma omp parallel private (i,j,mx)
    {
            
        double *in        = (double *) fftw_malloc(sizeof(double) * N); /********************* FFTW *********************/
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NC); /********************* FFTW *********************/

        double _Complex *y    = (double _Complex *) malloc(Ny * sizeof(double _Complex));
        fftw_plan plan; /********************* FFTW *********************/
            
    #pragma omp critical (make_plan)
        { plan = fftw_plan_dft_r2c_1d ( N, in, out, FFTW_ESTIMATE ); } /********************* FFTW *********************/

        forwardDST(sys, dst, sys.rhs, rhat, plan, in, out);
        
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
      
        reverseDST(sys, dst, xhat, sys.sol, plan, in, out);
            
        fftw_destroy_plan(plan); /********************* FFTW *********************/
        free(in); in = NULL;
        fftw_free(out); out = NULL; /********************* FFTW *********************/
        free(y); y = NULL;
    }

    free(rhat); rhat = NULL;
    free(xhat); xhat = NULL;
}
