#include "../headers/structs.h"
#include "../headers/prototypes.h"

#include "lu_2nd_kernels.h"

void LU( System sys ) {

#ifdef USE_CUFFTW
    create_2th_order_wrapper( sys );
#else
    int    Nx = sys.lat.Nx, Ny = sys.lat.Ny, Nz = sys.lat.Nz;
    double complex *a = sys.a, *b = sys.b, *c = sys.c, *d = sys.d;
    double complex *am = sys.am, *bm = sys.bm, *cm = sys.cm, *dm = sys.dm;
    double complex *ap = sys.ap, *bp = sys.bp, *cp = sys.cp, *dp = sys.dp;

    int     l, i, j, my, mxy, mxyz, Nxy, Nxz;
    double *eigenValue_x, *eigenValue_y;
    double complex *eigenValue_3D, *eigenValue_3DP, *eigenValue_3DM;
    /*
     Introduce the eigenvalues of the matrix on every layer
     */
    Nxy            = Nx * Ny;
    Nxz            = Nx * Nz;
    eigenValue_x   = malloc( Nx * sizeof( double ) );
    eigenValue_y   = malloc( Ny * sizeof( double ) );
    eigenValue_3D  = malloc( Nxy * Nz * sizeof( double complex ) );
    eigenValue_3DP = malloc( Nxy * Nz * sizeof( double complex ) );
    eigenValue_3DM = malloc( Nxy * Nz * sizeof( double complex ) );

    for ( i = 0; i < Nx; i++ ) {
        eigenValue_x[i] = 2.0 * cos( ( i + 1 ) * M_PI / ( Nx + 1 ) );
    }

    for ( j = 0; j < Ny; j++ ) {
        eigenValue_y[j] = 2.0 * cos( ( j + 1 ) * M_PI / ( Ny + 1 ) );
    }

    double eigenvalue_xy = 0;
    for ( j = 0; j < Ny; j++ ) {
        my = j * Nx;
        for ( i = 0; i < Nx; i++ ) {
            mxy           = i + my;
            eigenvalue_xy = eigenValue_x[i] * eigenValue_y[j];
            for ( l = 0; l < Nz; l++ ) {
                mxyz = mxy + l * Nxy; // l * Nxy + j * Nx + i
                eigenValue_3D[mxyz] = a[l] * eigenvalue_xy + b[l] * eigenValue_x[i] + c[l] * eigenValue_y[j] + d[l];
                eigenValue_3DP[mxyz] =
                    ap[l] * eigenvalue_xy + bp[l] * eigenValue_x[i] + cp[l] * eigenValue_y[j] + dp[l];
                eigenValue_3DM[mxyz] =
                    am[l] * eigenvalue_xy + bm[l] * eigenValue_x[i] + cm[l] * eigenValue_y[j] + dm[l];
            }
        }
    }
    /*
     Introduce the set of tridiagonal matrices
     */
    for ( j = 0; j < Ny; j++ ) {
        // my = j * Nxz;
        for ( i = 0; i < Nx; i++ ) {
            mxy         = i * Nz + j * Nxz;
            sys.L[mxy]  = 0.0;
            sys.U[mxy]  = 1.0 / eigenValue_3D[i + j * Nx];
            sys.Up[mxy] = eigenValue_3DP[i + j * Nx];
        }
    }

    for ( j = 0; j < Ny; j++ ) {
        // my = j * Nxz;
        for ( i = 0; i < Nx; i++ ) {
            mxy = i * Nz + j * Nxz;
            for ( l = 1; l < Nz; l++ ) {
                mxyz         = l + mxy;
                sys.L[mxyz]  = eigenValue_3DM[i + j * Nx + l * Nxy] * sys.U[l - 1 + i * Nz + j * Nxz];
                sys.U[mxyz]  = 1.0 / ( eigenValue_3D[i + j * Nx + l * Nxy] -
                                      sys.Up[l - 1 + i * Nz + j * Nxz] * sys.L[l + i * Nz + j * Nxz] );
                sys.Up[mxyz] = eigenValue_3DP[i + j * Nx + l * Nxy];
            }

        }
    }

    free( eigenValue_x );
    eigenValue_x = NULL;
    free( eigenValue_y );
    eigenValue_y = NULL;
    free( eigenValue_3D );
    eigenValue_3D = NULL;
#endif
}
