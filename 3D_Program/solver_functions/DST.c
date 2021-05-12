#include "../headers/structs.h"
#include "../headers/prototypes.h"

void DST(const int Nx, const int Ny, double *b_2D, double *bhat, fftw_plan p1, double *in1, fftw_complex *out1,fftw_plan p2,  double *in2, fftw_complex *out2)
{
    double coef;
    coef = 2.0/sqrt(Nx+1)/sqrt(Ny+1);
    
    int i, j, NR ,NC ;
    NR = 2*Nx + 2;
    NC = NR/2 + 1;

    for( j = 0; j < Ny; j++) {
        for( i = 0; i < Nx; i++){
            in1[i+1 + j*NR] = b_2D[i + j*Nx] ;
        }
    }
    
    fftw_execute(p1);
    
    NR = 2*Ny + 2;
    
    for( j = 0; j < Ny; j++) {
        for( i = 0; i < Nx; i++){
            b_2D[i + j*Nx] = cimag(out1[i+1 + j*NC]) ;
        }
    }
        

    for( j = 0; j < Ny; j++){
        for( i = 0; i < Nx; i++) {
            in2[j+1 + i*NR] = b_2D[i + j*Nx] ;
        }
    }
    
    NC = NR/2 + 1;

    fftw_execute(p2);


    for( i = 0; i < Nx; i++) {
        for( j = 0; j < Ny; j++){
            bhat[i + j*Nx] = cimag(out2[j+1 + i*NC]) ;
        }
    }
    

    for( j = 0; j < Ny; j++) {
        for( i = 0; i < Nx; i++){
            bhat[i + j*Nx] = coef * bhat[i + j*Nx] ;
        }
    }
    
    return ;
}




