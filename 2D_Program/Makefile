CC = gcc
CFLAGS = -Wall -fopenmp -O2
INCLUDES = -I/usr/local/include -I/usr/local/cuda/include
LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib64 
# LIBS = -lfftw3_omp -lfftw3 -lm -lnvToolsExt
# LIBS = -lcufft -lcufftw -lm -lnvToolsExt
SRCS = $(wildcard *.c) $(wildcard */*.c)
OBJS = $(SRCS:.c=.o)
MAIN = Direct_Solver

ifneq ($(USE_OMP), 0)
	CFLAGS += -USE_OMP
	LIBS = -lfftw3_omp -lfftw3 -lm -lnvToolsExt
else
	LIBS = -lcufft -lcufftw -lm -lnvToolsExt
endif

.PHONY: depend clean

all: $(MAIN)
	@echo  Program has been compiled

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN) */*.o

depend: $(SRCS)
	makedepend $(INCLUDES) $^
