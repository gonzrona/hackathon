#include "../headers/structs.h"
#include "../headers/prototypes.h"

#include "../solver_functions/cuda_helper.h"

System defineSystem(int argc, char **argv) {
    
    System sys = userInput();

    if (argc == 4) {
        if (atoi(argv[1]) == 6) { sys.order = sixth; }
        else if (atoi(argv[1]) == 4) { sys.order = fourth; }
        else { sys.order = second; }
        sys.lat.Nx = atoi(argv[2]);
        sys.lat.Ny = sys.lat.Nx;
        sys.threadCount = atoi(argv[3]);
    }
    else if (argc == 3) {
        if (atoi(argv[1]) == 6) { sys.order = sixth; }
        else if (atoi(argv[1]) == 4) { sys.order = fourth; }
        else { sys.order = second; }
        sys.lat.Nx = atoi(argv[2]);
        sys.lat.Ny = sys.lat.Nx;
        // printf("Enter number of OMP threads: "); scanf("%d", &sys.threadCount);
    }
    else if (argc == 2) {
        if (atoi(argv[1]) == 6) { sys.order = sixth; }
        else if (atoi(argv[1]) == 4) { sys.order = fourth; }
        else { sys.order = second; }
        // printf("Enter Nx: "); scanf("%d", &sys.lat.Nx);
        // printf("Enter Ny: "); scanf("%d", &sys.lat.Ny);
        // printf("Enter number of OMP threads: "); scanf("%d", &sys.threadCount);
    }
    else {
        // int order;
        // printf("Enter Order (2, 4, or 6): "); scanf("%d", &order);
        // if (order == 6) { sys.order = sixth; }
        // else if (order == 4) { sys.order = fourth; }
        // else { sys.order = second; }
        // printf("Enter Nx: "); scanf("%d", &sys.lat.Nx);
        // printf("Enter Ny: "); scanf("%d", &sys.lat.Ny);
        // printf("Enter number of OMP threads: "); scanf("%d", &sys.threadCount);
    }
    
    if (sys.order == sixth && sys.lat.Nx != sys.lat.Ny) {
        sys.lat.Ny = sys.lat.Nx;
        printf("WARNING: Sixth order requires a uniform grid size, Ny was set to Nx\n");
    }

    sys.lat.hx = (sys.lat.x1-sys.lat.x0)/(sys.lat.Nx+1);
    sys.lat.hy = (sys.lat.y1-sys.lat.y0)/(sys.lat.Ny+1);
    
    sys.lat.Nxy = sys.lat.Nx * sys.lat.Ny;

#if USE_CUFFTW
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.sol, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.sol_analytic, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.rhs, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.L, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.U, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.Up, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.res, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.error, sys.lat.Nxy * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.k, sys.lat.Nxy * sizeof(double _Complex), 1));

    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.bp, sys.lat.Ny * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.ap, sys.lat.Ny * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.b, sys.lat.Ny * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.a, sys.lat.Ny * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.bm, sys.lat.Ny * sizeof(double _Complex), 1));
    CUDA_RT_CALL(cudaMallocManaged((void **)&sys.am, sys.lat.Ny * sizeof(double _Complex), 1));

#else
    sys.sol             = malloc(sys.lat.Nxy * sizeof(double _Complex));
    sys.sol_analytic    = malloc(sys.lat.Nxy * sizeof(double _Complex));
    sys.rhs             = malloc(sys.lat.Nxy * sizeof(double _Complex));
    sys.L               = malloc(sys.lat.Nxy * sizeof(double _Complex));
    sys.U               = malloc(sys.lat.Nxy * sizeof(double _Complex));
    sys.Up              = malloc(sys.lat.Nxy * sizeof(double _Complex));
    sys.res             = malloc(sys.lat.Nxy * sizeof(double _Complex));
    sys.error           = malloc(sys.lat.Nxy * sizeof(double _Complex)); // can reuse res rather than allocating more memory
    sys.k               = malloc(sys.lat.Ny  * sizeof(double _Complex));
    
    sys.bp = malloc(sys.lat.Ny * sizeof(double _Complex));
    sys.ap = malloc(sys.lat.Ny * sizeof(double _Complex));
    sys.b  = malloc(sys.lat.Ny * sizeof(double _Complex));
    sys.a  = malloc(sys.lat.Ny * sizeof(double _Complex));
    sys.bm = malloc(sys.lat.Ny * sizeof(double _Complex));
    sys.am = malloc(sys.lat.Ny * sizeof(double _Complex));
#endif

    return sys;
}

void clearMemory(System sys) {
#if USE_CUFFTW
    CUDA_RT_CALL(cudaFree(sys.sol));
    CUDA_RT_CALL(cudaFree(sys.sol_analytic));
    CUDA_RT_CALL(cudaFree(sys.rhs));
    CUDA_RT_CALL(cudaFree(sys.L));
    CUDA_RT_CALL(cudaFree(sys.U));
    CUDA_RT_CALL(cudaFree(sys.Up));
    CUDA_RT_CALL(cudaFree(sys.res));
    CUDA_RT_CALL(cudaFree(sys.error));
    CUDA_RT_CALL(cudaFree(sys.k));

    CUDA_RT_CALL(cudaFree(sys.bp));
    CUDA_RT_CALL(cudaFree(sys.ap));
    CUDA_RT_CALL(cudaFree(sys.b));
    CUDA_RT_CALL(cudaFree(sys.a));
    CUDA_RT_CALL(cudaFree(sys.bm));
    CUDA_RT_CALL(cudaFree(sys.am));
#else
    free(sys.sol); sys.sol = NULL;
    free(sys.sol_analytic); sys.sol_analytic = NULL;
    free(sys.rhs); sys.rhs = NULL;
    free(sys.L); sys.L = NULL;
    free(sys.U); sys.U = NULL;
    free(sys.Up); sys.Up = NULL;
    free(sys.res); sys.res = NULL;
    free(sys.error); sys.error = NULL; // can reuse res rather than allocating more memory
    free(sys.k); sys.k = NULL;
    
    free(sys.bp); sys.bp = NULL;
    free(sys.ap); sys.ap = NULL;
    free(sys.b);  sys.b  = NULL;
    free(sys.a);  sys.a  = NULL;
    free(sys.bm); sys.bm = NULL;
    free(sys.am); sys.am = NULL;
#endif
}
