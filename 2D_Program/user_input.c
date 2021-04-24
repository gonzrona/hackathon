#include "headers/structs.h"
#include "headers/prototypes.h"

double _Complex k(double y) {
//    return y + I;
//    return y + I*y;
    return y*y + I*y;
}

double _Complex k2(double y) {
    return k(y)*k(y);
}

double _Complex k2_y(double y) {
//    return 2.0*y + 2.0*I;
//    return 4.0*I*y;
    return 4.0*y*y*y + 6.0*I*y*y - 2.0*y;
}

double _Complex k2_yy(double y) {
//    return 2.0;
//    return 4.0*I;
    return 12.0*y*y + 12.0*I*y - 2.0;
}

double analytic_solution(double x, double y) {
    return sin(x)*sin(y);
}

double _Complex f(double x, double y) {
    return (k(y)*k(y) - 2.0)*sin(x)*sin(y);
}

double _Complex f_y(double x, double y) {
    return (k(y)*k(y) - 2.0)*sin(x)*cos(y) + k2_y(y)*sin(x)*sin(y);
}

double _Complex f_xx(double x, double y) {
    return (2.0 - k(y)*k(y))*sin(x)*sin(y);
}

double _Complex f_yy(double x, double y) {
//    return (4.0 - y*y)*sin(x)*sin(y) + 4.0*y*sin(x)*cos(y);
    return (2.0 - k(y)*k(y) + k2_yy(y) )*sin(x)*sin(y) + 2.0*k2_y(y)*sin(x)*cos(y);
}

System userInput() {
    System sys;

//    DEFINE THE DOMAIN
//    sys.lat.x0 = 0.0; sys.lat.x1 = M_PI;
//    sys.lat.y0 = 0.0; sys.lat.y1 = M_PI;

    sys.lat.x0 = 0.5; sys.lat.x1 = M_PI - 0.5;
    sys.lat.y0 = 0.5; sys.lat.y1 = M_PI - 0.5;

    return sys;
}
