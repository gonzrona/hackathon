#include "headers/structs.h"
#include "headers/prototypes.h"


void background3D(System sys){
    int Nz = sys.lat.Nz;
    double hz = sys.lat.hz, z0 = sys.lat.z0;
    double complex *k_bg_ext = sys.k_bg_ext, *k2_bg_ext = sys.k2_bg_ext;
    double A = sys.A, B = sys.B, C = sys.C;
    
    int l;
    double z;
    for( l = 0; l < Nz+2; l++){
        z = z0 + hz*l ;
        k_bg_ext[l] = (A-B*sin(C*z));
        k2_bg_ext[l] = k_bg_ext[l]*k_bg_ext[l];
    }
}


System userInput() {
    System sys;

//    DEFINE THE DOMAIN
    sys.lat.x0 = 0.0; sys.lat.x1 = M_PI;
    sys.lat.y0 = 0.0; sys.lat.y1 = M_PI;
    sys.lat.z0 = 0.0; sys.lat.z1 = M_PI;
    
    /*
    Parameter used of the test problem with k(z) = A-B*sin(C*z)

    Note:
    For this code it is necessary that both beta and gamma have to be integer to satisfy the boundary problem
    */
    
    sys.A = 10.0;
    sys.B = 0.0;//9.0;
    sys.C = 10.0;
    sys.gamma = 6.0;//9.0;
    sys.beta = sqrt(sys.A*sys.A+sys.B*sys.B-sys.gamma*sys.gamma);
    

    return sys;
}
