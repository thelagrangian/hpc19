CC=gcc
CXX=g++
NC=nvcc
MC=mpicc
CFLAGS=-O3 -fopenmp -lm -march=native
CXXFLAGS=-O3 -fopenmp -march=native
NVOMP=-Xcompiler "-fopenmp"
SRCS := $(wildcard *.c)
EXEC := $(SRCS:%.c=%)


all: $(EXEC)

%: %.c
	$(MC) -o $@ $<

.PHONY: clean
clean:
	rm -f $(EXEC)
