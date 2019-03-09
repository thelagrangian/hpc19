CC=gcc
CXX=g++
CFLAGS=-O3 -fopenmp -lm -march=native
CXXFLAGS=-O3 -fopenmp -march=native
DEPS=utils.h
EXEC=MMult1 jacobi2D_omp gs2D_omp omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 val_test01_solved val_test02_solved


all: $(EXEC)


MMult1: MMult1.o
	$(CXX) -o $@ $< $(CXXFLAGS)

MMult1.o: MMult1.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

jacobi2D_omp: jacobi2D_omp.o
	$(CC) -o $@ $< $(CFLAGS)

jacobi2D.o: jacobi2D_omp.c
	$(CC) -c -o $@ $< $(CFLAGS)

gs2D_omp: gs2D_omp.o
	$(CC) -o $@ $< $(CFLAGS)

gs2D_omp.o: gs2D_omp.c
	$(CC) -c -o $@ $< $(CFLAGS)

omp_solved2: omp_solved2.o
	$(CC) -o $@ $< $(CFLAGS)

omp_solved3: omp_solved3.o
	$(CC) -o $@ $< $(CFLAGS)

omp_solved4: omp_solved4.o
	$(CC) -o $@ $< $(CFLAGS)

omp_solved5: omp_solved5.o
	$(CC) -o $@ $< $(CFLAGS)

omp_solved6: omp_solved6.o
	$(CC) -o $@ $< $(CFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

val_test01_solved: val_test01_solved.o
	$(CXX) -o $@ $< $(CXXFLAGS)

val_test02_solved: val_test02_solved.o
	$(CXX) -o $@ $< $(CXXFLAGS)

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f $(EXEC) *.o
