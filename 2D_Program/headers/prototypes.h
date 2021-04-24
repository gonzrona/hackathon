System userInput();
System defineSystem(int argc, char **argv);

double analytic_solution(double x, double y);
double _Complex f(double x, double y);
double _Complex k(double y);

double _Complex k2(double y);
double _Complex k2_y(double y);
double _Complex k2_yy(double y);
double _Complex f_y(double x, double y);
double _Complex f_xx(double x, double y);
double _Complex f_yy(double x, double y);

void k2_y_4th_order(System sys, double _Complex *k2_e, double _Complex *k2y);
void f_y_4th_order(System sys, double _Complex *f_e, double _Complex *fy);
void f_yy_4th_order(System sys, double _Complex *f_e, double _Complex *fyy_e);
void f_xx_4th_order(System sys, double _Complex *f_e, double _Complex *fxx_e);
void derivative_xx_2nd_order(System sys, double _Complex *f, double _Complex *fxx);
void derivative_yy_2nd_order(System sys, double _Complex *f, double _Complex *fyy);

void coefficients(System sys);
void rhs(System sys);
void LU(System sys);
void solver(System sys);
void residual(System sys);
double normInf(int n, double _Complex *x);

void clearMemory(System sys);

Time tic();
Time toc(Time time);

void printOrder(System sys);
void printLattice(Lattice lat);
void printDoubleArray(int n, double _Complex *a, char c[]);
