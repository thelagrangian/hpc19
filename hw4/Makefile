CC=gcc
CXX=g++
NC=nvcc
CFLAGS=-O3 -fopenmp -lm -march=native
CXXFLAGS=-O3 -fopenmp -march=native
NVOMP=-Xcompiler "-fopenmp"
SRCS := $(wildcard *.cu)
EXEC := $(SRCS:%.cu=%)


all: $(EXEC)

%: %.cu
	$(NC) $(NVOMP) -o $@ $<

.PHONY: clean
clean:
	rm -f $(EXEC)
