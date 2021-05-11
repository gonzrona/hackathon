NVCC = nvcc
CC = gcc
CFLAGS = -Xcompiler -Wall -Xcompiler -fopenmp -O2 --expt-relaxed-constexpr
INCLUDES = -I/usr/local/include -I/usr/local/cuda/include
LFLAGS = -L/usr/local/lib -L/usr/local/cuda/lib64 
LIBS = -lm -lnvToolsExt
SRCS = $(wildcard *.c) $(wildcard */*.c)
OBJS = $(SRCS:.c=.o)

CUSRCS = $(wildcard */*.cu)
CUOBJS = $(CUSRCS:.cu=.o)

ARCHES	:=-gencode arch=compute_70,code=\"compute_70,sm_70\" \
			-gencode arch=compute_75,code=\"compute_75,sm_75\" \
			-gencode arch=compute_80,code=\"compute_80,sm_80\"

NVCC_MAJOR = $(shell ${NVCC} --version | grep "release" | cut  -f6 -d' ' | cut -c2- | cut -f1 -d'.')
NVCC_MINOR = $(shell ${NVCC} --version | grep "release" | cut  -f6 -d' ' | cut -c2- | cut -f2 -d'.')

MAIN = Direct_Solver

ifeq ($(shell test $(NVCC_MAJOR) -ge 11; echo $$?),0)
ifeq ($(shell test $(NVCC_MINOR) -ge 2; echo $$?),0)
	THREADS = -t 0
endif
endif

ifneq ($(origin USE_CUFFTW), undefined)
	ifneq ($(USE_CUFFTW), 0)
		CFLAGS += -DUSE_CUFFTW
		LIBS += -lcufft -lcudart
	else
		LIBS += -lfftw3_omp -lfftw3
	endif
else
	LIBS += -lfftw3_omp -lfftw3
endif

ifneq ($(origin USE_COMBINE), undefined)
	ifneq ($(USE_COMBINE), 0)
		CFLAGS += -DUSE_COMBINE
	endif
endif

.PHONY: depend clean

all: $(MAIN)
	@echo  Program has been compiled

$(MAIN): $(OBJS) $(CUOBJS)
	$(NVCC) -ccbin $(CC) $(CFLAGS) $(INCLUDES) -o ${MAIN} ${OBJS} ${CUOBJS} $(LFLAGS) $(LIBS)

%.o: %.c
	$(NVCC) ${THREADS} -ccbin $(CC) $(CFLAGS) $(INCLUDES) ${ARCHES} -c $< -o $@

%.o: %.cu
	$(NVCC) ${THREADS} -ccbin $(CC) -std=c++11 -lineinfo $(CFLAGS) $(INCLUDES) ${ARCHES} -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN) */*.o

depend: $(SRCS)
	makedepend $(INCLUDES) $^
