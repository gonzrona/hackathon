# hackathon

## 2D_Program

Uses OpenMP, we have no purely sequential code.

### solver_functions 

The directory solver_functions contains two files, solver.c and DST.c. These two files contain the solver that we will focus on in the Hackathon.

All FFTW calls are contained in this directory. 

solver.c:
    contains FFTW functions on lines 19, 20, 25, 28, 67, and 69
    lines 36 and 61 call DST() which contains the FFTW plan execution
    
DST.c:
    contains FFTW functions on lines 11 and 17 ( FFT of real and complex parts respectively )

We store the matrices as 1D arrays:

Matrix:
[ a_{1,1}  a_{1,2}  a_{1,3}
  a_{2,1}  a_{2,2}  a_{2,3}
  a_{3,1}  a_{3,2}  a_{3,3} ]
  
  1D array:
  [ a_{1,1}  a_{1,2}  a_{1,3}  a_{2,1}  a_{2,2}  a_{2,3}  a_{3,1}  a_{3,2}  a_{3,3} ]
  
  
  
  ### Execution Instructions
  
  COMPILE:
  
       make
        
  TO RUN:

      ./Direct_Solver (order) (Nx=Ny) (threads)

  EXAMPLE:

      ./Direct_Solver 6 100 2

      6th order
      100x100 
      2 omp threads
