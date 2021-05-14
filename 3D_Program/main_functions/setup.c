
#include "../headers/structs.h"
#include "../headers/prototypes.h"

#include "../solver_functions/cuda_helper.h"

System defineSystem( int argc, char **argv ) {

    System sys = userInput( );

    sys.order  = second;
    sys.lat.Nx = 100;
    sys.lat.Ny = sys.lat.Nx;
    sys.lat.Nz = sys.lat.Nx;

    if ( argc == 5 ) {
        if ( atoi( argv[1] ) == 6 ) {
            sys.order = sixth;
        } else if ( atoi( argv[1] ) == 4 ) {
            sys.order = fourth;
        } else {
            sys.order = second;
        }
        sys.lat.Nx = atoi( argv[2] );
        sys.lat.Ny = atoi( argv[3] );
        sys.lat.Nz = atoi( argv[4] );
    } else if ( argc == 4 ) {
        if ( atoi( argv[1] ) == 6 ) {
            sys.order = sixth;
        } else if ( atoi( argv[1] ) == 4 ) {
            sys.order = fourth;
        } else {
            sys.order = second;
        }
        sys.lat.Nx = atoi( argv[2] );
        sys.lat.Ny = atoi( argv[3] );
        sys.lat.Nz = sys.lat.Nx;
    } else if ( argc == 3 ) {
        if ( atoi( argv[1] ) == 6 ) {
            sys.order = sixth;
        } else if ( atoi( argv[1] ) == 4 ) {
            sys.order = fourth;
        } else {
            sys.order = second;
        }
        sys.lat.Nx = atoi( argv[2] );
        sys.lat.Ny = sys.lat.Nx;
        sys.lat.Nz = sys.lat.Nx;
    } else if ( argc == 2 ) {
        if ( atoi( argv[1] ) == 6 ) {
            sys.order = sixth;
        } else if ( atoi( argv[1] ) == 4 ) {
            sys.order = fourth;
        } else {
            sys.order = second;
        }
    }

    sys.lat.hx = ( sys.lat.x1 - sys.lat.x0 ) / ( sys.lat.Nx + 1 );
    sys.lat.hy = ( sys.lat.y1 - sys.lat.y0 ) / ( sys.lat.Ny + 1 );
    sys.lat.hz = ( sys.lat.z1 - sys.lat.z0 ) / ( sys.lat.Nz + 1 );

    sys.lat.Nxyz = sys.lat.Nx * sys.lat.Ny * sys.lat.Nz;

#if USE_CUFFTW
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.a, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.b, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.c, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.d, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.ap, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.bp, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.cp, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.dp, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.am, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.bm, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.cm, sys.lat.Nz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.dm, sys.lat.Nz * sizeof( double complex ), 1 ));

    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.k_bg_ext, ( sys.lat.Nz + 2 ) * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.k2_bg_ext, ( sys.lat.Nz + 2 ) * sizeof( double complex ), 1 ));

    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.sol_analytic, sys.lat.Nxyz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.rhs, sys.lat.Nxyz * sizeof( double complex ), 1 ));

    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.L, sys.lat.Nxyz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.U, sys.lat.Nxyz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.Up, sys.lat.Nxyz * sizeof( double complex ), 1 ));

    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.sol, sys.lat.Nxyz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.res, sys.lat.Nxyz * sizeof( double complex ), 1 ));
    CUDA_RT_CALL( cudaMallocManaged( ( void ** )&sys.error, sys.lat.Nxyz * sizeof( double complex ), 1 ));
#else
    sys.a  = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.b  = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.c  = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.d  = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.ap = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.bp = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.cp = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.dp = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.am = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.bm = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.cm = malloc( sys.lat.Nz * sizeof( double complex ) );
    sys.dm = malloc( sys.lat.Nz * sizeof( double complex ) );

    sys.k_bg_ext  = malloc( ( sys.lat.Nz + 2 ) * sizeof( double complex ) );
    sys.k2_bg_ext = malloc( ( sys.lat.Nz + 2 ) * sizeof( double complex ) );

    sys.sol_analytic = malloc( sys.lat.Nxyz * sizeof( double complex ) );
    sys.rhs          = malloc( sys.lat.Nxyz * sizeof( double complex ) );

    sys.L  = malloc( sys.lat.Nxyz * sizeof( double complex ) );
    sys.U  = malloc( sys.lat.Nxyz * sizeof( double complex ) );
    sys.Up = malloc( sys.lat.Nxyz * sizeof( double complex ) );

    sys.sol   = malloc( sys.lat.Nxyz * sizeof( double complex ) );
    sys.res   = malloc( sys.lat.Nxyz * sizeof( double complex ) );
    sys.error = malloc( sys.lat.Nxyz * sizeof( double complex ) );
#endif

    return sys;
}

void clearMemory( System sys ) {
#if USE_CUFFTW
    CUDA_RT_CALL( cudaFree( sys.a ));
    sys.a = NULL;
    CUDA_RT_CALL( cudaFree( sys.b ));
    sys.b = NULL;
    CUDA_RT_CALL( cudaFree( sys.c ));
    sys.c = NULL;
    CUDA_RT_CALL( cudaFree( sys.d ));
    sys.d = NULL;
    CUDA_RT_CALL( cudaFree( sys.ap ));
    sys.ap = NULL;
    CUDA_RT_CALL( cudaFree( sys.bp ));
    sys.bp = NULL;
    CUDA_RT_CALL( cudaFree( sys.cp ));
    sys.cp = NULL;
    CUDA_RT_CALL( cudaFree( sys.dp ));
    sys.dp = NULL;
    CUDA_RT_CALL( cudaFree( sys.am ));
    sys.am = NULL;
    CUDA_RT_CALL( cudaFree( sys.bm ));
    sys.bm = NULL;
    CUDA_RT_CALL( cudaFree( sys.cm ));
    sys.cm = NULL;
    CUDA_RT_CALL( cudaFree( sys.dm ));
    sys.dm = NULL;

    CUDA_RT_CALL( cudaFree( sys.k_bg_ext ));
    sys.k_bg_ext = NULL;
    CUDA_RT_CALL( cudaFree( sys.k2_bg_ext ));
    sys.k2_bg_ext = NULL;

    CUDA_RT_CALL( cudaFree( sys.sol_analytic ));
    sys.sol_analytic = NULL;
    CUDA_RT_CALL( cudaFree( sys.rhs ));
    sys.rhs = NULL;

    CUDA_RT_CALL( cudaFree( sys.L ));
    sys.L = NULL;
    CUDA_RT_CALL( cudaFree( sys.U ));
    sys.U = NULL;
    CUDA_RT_CALL( cudaFree( sys.Up ));
    sys.Up = NULL;

    CUDA_RT_CALL( cudaFree( sys.sol ));
    sys.sol = NULL;
    CUDA_RT_CALL( cudaFree( sys.res ));
    sys.res = NULL;
    CUDA_RT_CALL( cudaFree( sys.error ));
    sys.error = NULL;
#else
    free( sys.a );
    sys.a = NULL;
    free( sys.b );
    sys.b = NULL;
    free( sys.c );
    sys.c = NULL;
    free( sys.d );
    sys.d = NULL;
    free( sys.ap );
    sys.ap = NULL;
    free( sys.bp );
    sys.bp = NULL;
    free( sys.cp );
    sys.cp = NULL;
    free( sys.dp );
    sys.dp = NULL;
    free( sys.am );
    sys.am = NULL;
    free( sys.bm );
    sys.bm = NULL;
    free( sys.cm );
    sys.cm = NULL;
    free( sys.dm );
    sys.dm = NULL;

    free( sys.k_bg_ext );
    sys.k_bg_ext = NULL;
    free( sys.k2_bg_ext );
    sys.k2_bg_ext = NULL;

    free( sys.sol_analytic );
    sys.sol_analytic = NULL;
    free( sys.rhs );
    sys.rhs = NULL;

    free( sys.L );
    sys.L = NULL;
    free( sys.U );
    sys.U = NULL;
    free( sys.Up );
    sys.Up = NULL;

    free( sys.sol );
    sys.sol = NULL;
    free( sys.res );
    sys.res = NULL;
    free( sys.error );
    sys.error = NULL;
#endif
}
