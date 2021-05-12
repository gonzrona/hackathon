#include "../headers/structs.h"
#include "../headers/prototypes.h"

int main(int argc, char **argv){
    
    System sys = defineSystem(argc, argv);
    
    printOrder(sys);
    printf("\tGrid %d x %d x %d\n", sys.lat.Nx, sys.lat.Ny, sys.lat.Nz);
    
    
    background3D(sys);

    coefficients(sys);

    rhs(sys);
    
    LU(sys);
    
    Time time = tic();
    solver(sys);
    time = toc(time);
    
    printf("\n \tSolver time = %f sec \n", time.computed_time );
//    printf("   \tSolver wall-time = %f sec \n", time.computed_t);
    printf("   \tSolver wall-time (sys/time) = %f sec \n\n", time.computed_t_n);
    
    
    residual(sys);

    printf("\t||error||_inf =  %10.7e \n\n",normInf(sys.lat.Nxyz, sys.error));
    printf("\t||res||_inf =  %10.7e \n\n",normInf(sys.lat.Nxyz, sys.res));
    printf("\t||res||_2 =  %10.7e \n\n",normL2(sys.lat.Nxyz, sys.res));
    
    
    clearMemory(sys);

    return 0;
}
