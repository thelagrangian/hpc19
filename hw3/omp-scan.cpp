#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  int nths_set = 8;
  int idx_trigger = 0;
#pragma omp parallel num_threads(nths_set)
{
  int num_threads = omp_get_num_threads();
  int tid         = omp_get_thread_num();
  int length      = (n + num_threads - 1)/num_threads;
  int beginidx    = length * tid;
  int endidx      = beginidx + length;
  long correction  = 0;

  //printf("%d/%d\n", tid, num_threads);

  if(endidx>n)
    endidx = n;

  if(tid == 0)
    prefix_sum[beginidx] = 0;
  else
    prefix_sum[beginidx] = A[beginidx-1];

  for(int i=beginidx + 1; i<endidx; ++i)
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];

#pragma omp barrier

  for(int i=beginidx-1; i>0; i-=length)
    correction += prefix_sum[i];

#pragma omp barrier
  for(int i=beginidx; i<endidx; ++i)
    prefix_sum[i] += correction;
}

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  srand(time(NULL));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
