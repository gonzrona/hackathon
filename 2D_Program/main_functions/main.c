#include "../headers/structs.h"
#include "../headers/prototypes.h"

int main (int argc, char **argv) {
    
    System sys = defineSystem(argc, argv);
    omp_set_num_threads(sys.threadCount);

    printOrder(sys);
    printf("\tGrid %d x %d\n", sys.lat.Nx, sys.lat.Ny);
    printf("\tOpenMP Threads: %d\n", sys.threadCount);
        
    Time time = tic();

    coefficients(sys);
            
    rhs(sys);
    
    LU(sys);
    
    time = toc(time);
    
    printf("\n \tSet up time = %f sec \n", time.computed_time );
    printf("   \tSet up wall-time = %f sec \n", time.computed_t);
    printf("   \tSet up wall-time (sys/time) = %f sec \n\n", time.computed_t_n);


    time = tic();
    
    solver(sys);
    
    time = toc(time);
    
    printf("\n \tSolver time = %f sec \n", time.computed_time );
    printf("   \tSolver wall-time = %f sec \n", time.computed_t);
    printf("   \tSolver wall-time (sys/time) = %f sec \n\n", time.computed_t_n);

    residual(sys);
           
    printf("\t||res||_inf =  %10.7e \n\n",normInf(sys.lat.Nxy, sys.res));
    printf("\t||error||_inf =  %10.7e \n\n",normInf(sys.lat.Nxy, sys.error));

    clearMemory(sys);
    
    return 0;
}
