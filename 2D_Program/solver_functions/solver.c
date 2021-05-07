#include "../headers/structs.h"

void DST(enum Direction dir, System sys, DSTN dst, double _Complex *rhs, double _Complex *rhat);

void solver(System sys) {
    
    int i,j,mx;
    int Nx = sys.lat.Nx, Ny = sys.lat.Ny;
    int Nxy = Nx*Ny;
    int N = 2*Nx + 2;
    int NC = (N/2) + 1;
    
    DSTN dst;

    dst.Nx = Nx; dst.N = N; dst.Ny = Ny; dst.coef = sqrt(2.0/(Nx+1)); dst.NC = NC;
    
    const int rank = 1;
    int n[] = {N};
    const int howmany = Ny;
    int *inembed = n, *onembed = n;
    const int istride = 1; const int ostride = 1;
    const int idist = N; const int odist = NC;
    
    double *in1 = (double *) fftw_malloc(sizeof(double) * N*Ny);
    double *in2 = (double *) fftw_malloc(sizeof(double) * N*Ny);
    double _Complex *y    = (double _Complex *) malloc(Ny * sizeof(double _Complex));
    double _Complex *rhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    double _Complex *xhat = (double _Complex *) malloc(Nxy * sizeof(double _Complex));
    fftw_complex *out1 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * NC*Ny);
    fftw_complex *out2 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * NC*Ny);

    fftw_plan plan1 = fftw_plan_many_dft_r2c(rank, n, howmany, in1, inembed, istride, idist,  out1, onembed, ostride, odist, FFTW_ESTIMATE);
    fftw_plan plan2 = fftw_plan_many_dft_r2c(rank, n, howmany, in2, inembed, istride, idist,  out2, onembed, ostride, odist, FFTW_ESTIMATE);
    
    dst.in1 = in1; dst.in2 = in2;
    dst.out1 = out1; dst.out2 = out2;
    dst.plan1 = plan1; dst.plan2 = plan2;

    DST(forward, sys, dst, sys.rhs, rhat);
    
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
    
    DST(reverse, sys, dst, xhat, sys.sol);

    
    free(y); y = NULL;
    free(rhat); rhat = NULL;
    free(xhat); xhat = NULL;
    free(in1); in1 = NULL;
    fftw_free(out1); out1 = NULL;
    free(in2); in2 = NULL;
    fftw_free(out2); out2 = NULL;

}
