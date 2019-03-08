CC=gcc
CXX=g++
CFLAGS=-O3 -fopenmp -lm -march=native
DEPS=utils.h


all: MMult1

MMult1: MMult1.o
	$(CXX) -o $@ $< $(CFLAGS)

MMult1.o: MMult1.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)
