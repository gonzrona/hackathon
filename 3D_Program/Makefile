CC = gcc-10
CFLAGS = -Wall -g -fopenmp -O0
INCLUDES = -I/usr/local/include
LFLAGS = -L/usr/local/lib
LIBS = -lfftw3_omp -lfftw3 -lm
SRCS = $(wildcard *.c) $(wildcard */*.c)
OBJS = $(SRCS:.c=.o)
MAIN = Direct_Solver

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
