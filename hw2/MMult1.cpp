// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "utils.h"

#define BLOCK_SIZE 16

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}


void MMult_omp(long m, long n, long k, double *a, double *b, double *c)
{
  // TODO: See instructions below
  // Sequential algorithm: tiling techneque to optimize cache usage
  int num_iblock = (m+BLOCK_SIZE-1)/BLOCK_SIZE;
  int num_jblock = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
  int num_pblock = (k+BLOCK_SIZE-1)/BLOCK_SIZE;

#pragma omp parallel for collapse(2) num_threads(16)
  for(int ii = 0; ii< num_iblock; ++ii)
  {
    for(int jj=0; jj< num_jblock; ++jj)
    {
      int ilimit = BLOCK_SIZE;
      int jlimit = BLOCK_SIZE;
      double ar[BLOCK_SIZE*BLOCK_SIZE],br[BLOCK_SIZE*BLOCK_SIZE],cr[BLOCK_SIZE*BLOCK_SIZE];
      if(ii==num_iblock-1)
        ilimit = m - (num_iblock-1)*BLOCK_SIZE;
      if(jj==num_jblock-1)
        jlimit = n - (num_jblock-1)*BLOCK_SIZE;
      //bringing the current c matrix block into fast memory
      for(int i=0;i<ilimit;++i)
        for(int j=0;j<jlimit;++j)
          cr[i + j*BLOCK_SIZE] = c[BLOCK_SIZE*ii + i + (BLOCK_SIZE*jj + j)*m];
      for(int pp=0; pp<num_pblock;++pp)
      {
        int plimit = BLOCK_SIZE;
        if(pp==num_pblock-1)
          plimit = k - (num_pblock-1)*BLOCK_SIZE;
        //bringing the current a and b matrix blocks into fast memory
        for(int i=0;i<ilimit;++i)
          for(int p=0;p<plimit;++p)
            ar[i+p*BLOCK_SIZE] = a[BLOCK_SIZE*ii + i+(pp*BLOCK_SIZE+p)*m];
        for(int p=0;p<plimit;++p)
          for(int j=0;j<jlimit;++j)
            br[p+j*BLOCK_SIZE] = b[pp*BLOCK_SIZE+p+(BLOCK_SIZE*jj + j)*k];
        //matrix block multiplication: pij pattern
        for(int p=0;p<plimit;++p)
          for(int i=0;i<ilimit;++i)
            for(int j=0;j<jlimit;++j)
              cr[i+j*BLOCK_SIZE] += ar[i+p*BLOCK_SIZE]*br[p+j*BLOCK_SIZE];
      }

      for(int i=0;i<ilimit;++i)
        for(int j=0;j<jlimit;++j)
          c[BLOCK_SIZE*ii + i + (BLOCK_SIZE*jj + j)*m]=cr[i + j*BLOCK_SIZE];

   }
 }


}




void MMult1(long m, long n, long k, double *a, double *b, double *c)
{
  // TODO: See instructions below
  // Sequential algorithm: tiling techneque to optimize cache usage
  int num_iblock = (m+BLOCK_SIZE-1)/BLOCK_SIZE;
  int num_jblock = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
  int num_pblock = (k+BLOCK_SIZE-1)/BLOCK_SIZE;

  double ar[BLOCK_SIZE*BLOCK_SIZE],br[BLOCK_SIZE*BLOCK_SIZE],cr[BLOCK_SIZE*BLOCK_SIZE];

  for(int ii = 0; ii< num_iblock; ++ii)
  {
    for(int jj=0; jj< num_jblock; ++jj)
    {
      int ilimit = BLOCK_SIZE;
      int jlimit = BLOCK_SIZE;
      if(ii==num_iblock-1)
        ilimit = m - (num_iblock-1)*BLOCK_SIZE;
      if(jj==num_jblock-1)
        jlimit = n - (num_jblock-1)*BLOCK_SIZE;
      //bringing the current c matrix block into fast memory
      for(int i=0;i<ilimit;++i)
        for(int j=0;j<jlimit;++j)
          cr[i + j*BLOCK_SIZE] = c[BLOCK_SIZE*ii + i + (BLOCK_SIZE*jj + j)*m];
      for(int pp=0; pp<num_pblock;++pp)
      {
        int plimit = BLOCK_SIZE;
        if(pp==num_pblock-1)
          plimit = k - (num_pblock-1)*BLOCK_SIZE;
        //bringing the current a and b matrix blocks into fast memory
        for(int i=0;i<ilimit;++i)
          for(int p=0;p<plimit;++p)
            ar[i+p*BLOCK_SIZE] = a[BLOCK_SIZE*ii + i+(pp*BLOCK_SIZE+p)*m];
        for(int p=0;p<plimit;++p)
          for(int j=0;j<jlimit;++j)
            br[p+j*BLOCK_SIZE] = b[pp*BLOCK_SIZE+p+(BLOCK_SIZE*jj + j)*k];
        //matrix block multiplication: pij pattern
        for(int p=0;p<plimit;++p)
          for(int i=0;i<ilimit;++i)
            for(int j=0;j<jlimit;++j)
              cr[i+j*BLOCK_SIZE] += ar[i+p*BLOCK_SIZE]*br[p+j*BLOCK_SIZE];
      }

      for(int i=0;i<ilimit;++i)
        for(int j=0;j<jlimit;++j)
          c[BLOCK_SIZE*ii + i + (BLOCK_SIZE*jj + j)*m]=cr[i + j*BLOCK_SIZE];

   }
 }


}


int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error        BlockTime\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;

    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    //double* a = (double*)     malloc(m * k * sizeof(double)); // m x k
    //double* b = (double*)     malloc(k * n * sizeof(double)); // k x n
    //double* c = (double*)     malloc(m * n * sizeof(double)); // m x n
    //double* c_ref = (double*) malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;


    // Added: another way to calculated time lapse
    struct timespec t1, t2;
    double timediff0;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult1(m, n, k, a, b, c_ref);
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    timediff0 = (t2.tv_sec + t2.tv_nsec/1e9) - (t1.tv_sec + t1.tv_nsec/1e9);
    //printf("omp: %10f\n",timediff0);



    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult_omp(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = m*n*(2.0*k-1)*NREPEATS/time/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = 4.0*m*n*k*NREPEATS/time/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e %10f\n", max_err,timediff0);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
    aligned_free(c_ref);

    //free(a);
    //free(b);
    //free(c);
    //free(c_ref);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
