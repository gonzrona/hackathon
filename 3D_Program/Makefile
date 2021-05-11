# Define the C compiler to use

CC = gcc

# define any compile-time flags
CFLAGS = -Wall -g -O3

# define any directories containing header files other than /usr/include
#
INCLUDES = -I/usr/local/include

# define library paths in addition to /usr/lib
#
LFLAGS = -L/usr/local/lib

# define any libraries to link into executable:
#
LIBS = -lfftw3 -lm

# define the C source files
SRCS = Direct_solver.c background3D.c second_test3D.c LU.c solver_3D_dir_Z_DDD.c myDST_2D.c residual3D.c normL2.c coefficients.c 2nd_order_partials.c 4th_order_partials.c

# define the C object files 
#
OBJS = $(SRCS:.c=.o)

# define the executable file 
MAIN = Direct_solver


#
# The following part of the makefile is generic; it can be used to 
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all:    $(MAIN)
	@echo  Simple compiler named mycc has been compiled

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
