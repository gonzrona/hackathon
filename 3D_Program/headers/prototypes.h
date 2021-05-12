System defineSystem(int argc, char **argv);
System userInput();

void printOrder(System sys);
void printLattice(Lattice lat);

void background3D(System sys);
void coefficients(System sys);
void rhs(System sys);
void LU(System sys);
void solver(System sys);
void residual(System sys);

double normInf(int n, double complex *x);
double normL2(int n, double complex *x);

void clearMemory(System sys);


Time tic();
Time toc(Time time);

double cpuSecond(void) ;
