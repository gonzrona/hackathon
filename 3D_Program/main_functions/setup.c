#include "../headers/structs.h"
#include "../headers/prototypes.h"

#include "../solver_functions/cuda_helper.h"

System defineSystem(int argc, char **argv) {
    
    System sys = userInput();
    
    sys.order = second;
    sys.lat.Nx = 100;
    sys.lat.Ny = sys.lat.Nx;
    sys.lat.Nz = sys.lat.Nx;

    if (argc == 5) {
        if (atoi(argv[1]) == 6) { sys.order = sixth; }
        else if (atoi(argv[1]) == 4) { sys.order = fourth; }
        else { sys.order = second; }
        sys.lat.Nx = atoi(argv[2]);
        sys.lat.Ny = atoi(argv[3]);
        sys.lat.Nz = atoi(argv[4]);
    }
    else if (argc == 4) {
        if (atoi(argv[1]) == 6) { sys.order = sixth; }
        else if (atoi(argv[1]) == 4) { sys.order = fourth; }
        else { sys.order = second; }
        sys.lat.Nx = atoi(argv[2]);
        sys.lat.Ny = atoi(argv[3]);
        sys.lat.Nz = sys.lat.Nx;
    }    
    else if (argc == 3) {
        if (atoi(argv[1]) == 6) { sys.order = sixth; }
        else if (atoi(argv[1]) == 4) { sys.order = fourth; }
        else { sys.order = second; }
        sys.lat.Nx = atoi(argv[2]);
        sys.lat.Ny = sys.lat.Nx;
        sys.lat.Nz = sys.lat.Nx;
    }
    else if (argc == 2) {
        if (atoi(argv[1]) == 6) { sys.order = sixth; }
        else if (atoi(argv[1]) == 4) { sys.order = fourth; }
        else { sys.order = second; }
    }
    
    sys.lat.hx = (sys.lat.x1-sys.lat.x0)/(sys.lat.Nx+1);
    sys.lat.hy = (sys.lat.y1-sys.lat.y0)/(sys.lat.Ny+1);
    sys.lat.hz = (sys.lat.z1-sys.lat.z0)/(sys.lat.Nz+1);

    sys.lat.Nxyz = sys.lat.Nx * sys.lat.Ny * sys.lat.Nz;

#if USE_CUFFTW
    cudaMallocManaged((void**)&sys.a, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.b, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.c, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.d, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.ap, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.bp, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.cp, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.dp, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.am, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.bm, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.cm, sys.lat.Nz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.dm, sys.lat.Nz * sizeof(double complex), 1);
    
    cudaMallocManaged((void**)&sys.k_bg_ext, (sys.lat.Nz+2) * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.k2_bg_ext, (sys.lat.Nz+2) * sizeof(double complex), 1);
    
    cudaMallocManaged((void**)&sys.sol_analytic, sys.lat.Nxyz * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.rhs, sys.lat.Nxyz * sizeof(double complex), 1);
    
    cudaMallocManaged((void**)&sys.L, sys.lat.Nxyz  * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.U, sys.lat.Nxyz  * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.Up, sys.lat.Nxyz  * sizeof(double complex), 1);

    cudaMallocManaged((void**)&sys.sol, sys.lat.Nxyz  * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.res, sys.lat.Nxyz  * sizeof(double complex), 1);
    cudaMallocManaged((void**)&sys.error, sys.lat.Nxyz  * sizeof(double complex), 1);
#else    
    sys.a = malloc(sys.lat.Nz * sizeof(double complex));
    sys.b = malloc(sys.lat.Nz * sizeof(double complex));
    sys.c = malloc(sys.lat.Nz * sizeof(double complex));
    sys.d = malloc(sys.lat.Nz * sizeof(double complex));
    sys.ap = malloc(sys.lat.Nz * sizeof(double complex));
    sys.bp = malloc(sys.lat.Nz * sizeof(double complex));
    sys.cp = malloc(sys.lat.Nz * sizeof(double complex));
    sys.dp = malloc(sys.lat.Nz * sizeof(double complex));
    sys.am = malloc(sys.lat.Nz * sizeof(double complex));
    sys.bm = malloc(sys.lat.Nz * sizeof(double complex));
    sys.cm = malloc(sys.lat.Nz * sizeof(double complex));
    sys.dm = malloc(sys.lat.Nz * sizeof(double complex));
    
    sys.k_bg_ext = malloc((sys.lat.Nz+2) * sizeof(double complex));
    sys.k2_bg_ext = malloc((sys.lat.Nz+2) * sizeof(double complex));
    
    sys.sol_analytic  = malloc(sys.lat.Nxyz * sizeof(double complex));
    sys.rhs = malloc(sys.lat.Nxyz * sizeof(double complex));
    
    sys.L    = malloc(sys.lat.Nxyz  * sizeof(double complex));
    sys.U    = malloc(sys.lat.Nxyz  * sizeof(double complex));
    sys.Up    = malloc(sys.lat.Nxyz  * sizeof(double complex));

    sys.sol    = malloc(sys.lat.Nxyz  * sizeof(double complex));
    sys.res    = malloc(sys.lat.Nxyz  * sizeof(double complex));
    sys.error    = malloc(sys.lat.Nxyz  * sizeof(double complex));
#endif

    return sys;
}

void clearMemory(System sys) {
#if USE_CUFFTW
    cudaFree(sys.a); sys.a = NULL;
    cudaFree(sys.b); sys.b = NULL;
    cudaFree(sys.c); sys.c = NULL;
    cudaFree(sys.d); sys.d = NULL;
    cudaFree(sys.ap); sys.ap = NULL;
    cudaFree(sys.bp); sys.bp = NULL;
    cudaFree(sys.cp); sys.cp = NULL;
    cudaFree(sys.dp); sys.dp = NULL;
    cudaFree(sys.am); sys.am = NULL;
    cudaFree(sys.bm); sys.bm = NULL;
    cudaFree(sys.cm); sys.cm = NULL;
    cudaFree(sys.dm); sys.dm = NULL;

    cudaFree(sys.k_bg_ext); sys.k_bg_ext = NULL;
    cudaFree(sys.k2_bg_ext); sys.k2_bg_ext = NULL;

    cudaFree(sys.sol_analytic); sys.sol_analytic = NULL;
    cudaFree(sys.rhs); sys.rhs = NULL;
    
    cudaFree(sys.L); sys.L = NULL;
    cudaFree(sys.U); sys.U = NULL;
    cudaFree(sys.Up); sys.Up = NULL;

    cudaFree(sys.sol); sys.sol = NULL;
    cudaFree(sys.res); sys.res = NULL;
    cudaFree(sys.error); sys.error = NULL;
#else   
    free(sys.a); sys.a = NULL;
    free(sys.b); sys.b = NULL;
    free(sys.c); sys.c = NULL;
    free(sys.d); sys.d = NULL;
    free(sys.ap); sys.ap = NULL;
    free(sys.bp); sys.bp = NULL;
    free(sys.cp); sys.cp = NULL;
    free(sys.dp); sys.dp = NULL;
    free(sys.am); sys.am = NULL;
    free(sys.bm); sys.bm = NULL;
    free(sys.cm); sys.cm = NULL;
    free(sys.dm); sys.dm = NULL;

    free(sys.k_bg_ext); sys.k_bg_ext = NULL;
    free(sys.k2_bg_ext); sys.k2_bg_ext = NULL;

    free(sys.sol_analytic); sys.sol_analytic = NULL;
    free(sys.rhs); sys.rhs = NULL;
    
    free(sys.L); sys.L = NULL;
    free(sys.U); sys.U = NULL;
    free(sys.Up); sys.Up = NULL;

    free(sys.sol); sys.sol = NULL;
    free(sys.res); sys.res = NULL;
    free(sys.error); sys.error = NULL;
#endif
}
