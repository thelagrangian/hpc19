#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<cuda.h>
#include<time.h>
#include<math.h>

#define BLOCK_SIZE (1<<10)
#define BLOCK_DIM  (1<<5)

//it is always c = a x b;

double dotproduct_omp(double*a, double*b, int n)
{
  double rval = 0.0;
  int i;
#pragma omp parallel for reduction(+:rval)
  for(i = 0; i < n; ++i)
    rval += a[i]*b[i];
  return rval;
}

void matrixarray_omp(double*c, double*a, double*b, int m, int n)
{                  //mx1       mxn       nx1
  int i;
#pragma omp parallel for
  for(i=0; i<m; ++i)
  {
    double rval = 0;
    int j;
    for(j=0; j<n; ++j)
      rval += a[i*n + j]*b[j];
    c[i] = rval;
  }
}

void dotproduct_cpu(double *a, double *b, double *c, int m, int n)
{
  struct timespec t0, t1;
  double time;
  double bandwidth;
  int i;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for(i = 0; i < m; ++i)
    c[i] = dotproduct_omp(a + i*n, b, n);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  time = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
  bandwidth = sizeof(double)*(2.0*m*n + m)/1e9/time;
  printf("cpu adapted dot product time spent: %f s, and cpu memory bandwidth: %f GB/s.\n", time, bandwidth);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  matrixarray_omp(c, a, b, m, n);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  time = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
  bandwidth = sizeof(double)*(2.0*m*n + m)/1e9/time;
  printf("cpu matrix-array product time spent: %f s, and cpu memory bandwidth: %f GB/s.\n", time, bandwidth);
}

__global__ void dotproduct_kernel(double *sum, double* a, double*b, int n)
{
  __shared__ double smem[BLOCK_SIZE];
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = BLOCK_SIZE/2;

  if(gidx < n)
    smem[threadIdx.x] = a[gidx]*b[gidx];
  else
    smem[threadIdx.x] = 0.0;
  __syncthreads();

  //stride = 512;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  __syncthreads();

  stride /= 2;
  //stride = 256;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  __syncthreads();

  stride /= 2;
  //stride = 128;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  __syncthreads();

  stride /= 2;
  //stride = 64;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  __syncthreads();

  stride /= 2;
  //stride = 32;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  __syncthreads();

  stride /= 2;
  //stride =16;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  //__syncwarp();
  __syncthreads();

  stride /= 2;
  //stride = 8;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  //__syncwarp();
  __syncthreads();

  stride /= 2;
  //stride = 4;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  //__syncwarp();
  __syncthreads();

  stride /= 2;
  //stride = 2;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  //__syncwarp();
  __syncthreads();

  stride /= 2;
  //stride = 1;
  if(threadIdx.x < stride)
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  //__syncwarp();
  __syncthreads();

  if(threadIdx.x ==0 )
    sum[blockIdx.x] = smem[0];
}

__global__ void matrixarray_kernel(double *c_d, double* a_d, double*b_d, int m, int n)
{
  int gidx = threadIdx.x + blockIdx.x + blockDim.x;
  double rval = 0.0;
  int i;
  for(i=0; i<n; ++i)
    rval += a_d[gidx*n + i] * b_d[i];
  c_d[gidx] = rval;
}

void dotproduct_gpu(double*a, double*b, double*c, int m, int n)
{
  struct timespec t0, t1;
  double time = 0.0;
  double bandwidth;
  double *a_d, *b_d, *sum, *sum_d, *ae_d, *c_d;
  int i, j;


  cudaMalloc(&a_d,   n*sizeof(double));
  cudaMalloc(&b_d,   n*sizeof(double));
  cudaMalloc(&sum_d, (n + BLOCK_SIZE - 1)/BLOCK_SIZE*sizeof(double));
  //sum = (double*)aligned_alloc(sizeof(double), num_blocks*sizeof(double));
  cudaMallocHost(&sum, (n + BLOCK_SIZE - 1)/BLOCK_SIZE*sizeof(double));
  cudaMalloc(&ae_d,m*n*sizeof(double));
  cudaMalloc(&c_d,   m*sizeof(double));

  cudaMemcpy(ae_d,a, m*n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b,   n*sizeof(double), cudaMemcpyHostToDevice);

  for(i = 0; i < m; ++i)
  {
    double c_i = 0.0;
    cudaMemcpy(a_d, a + i*n, n*sizeof(double), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    dotproduct_kernel<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(sum_d, a_d, b_d, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    time += (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
    cudaMemcpy(sum, sum_d, (n + BLOCK_SIZE - 1)/BLOCK_SIZE*sizeof(double), cudaMemcpyDeviceToHost);
    for(j = 0; j < (n + BLOCK_SIZE - 1)/BLOCK_SIZE; ++j)
      c_i += sum[j];
    c[i] = c_i;
  }
  bandwidth = sizeof(double)*(2*m*n)/1e9/time;
  printf("gpu adapted dot product time spent: %f s, and gpu memory bandwidth: %f GB/s.\n", time, bandwidth);

  clock_gettime(CLOCK_MONOTONIC, &t0);
  matrixarray_kernel<<<(m + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(c_d, ae_d, b_d, m, n);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  time = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
  bandwidth = sizeof(double)*(2*m*n)/1e9/time;
  printf("gpu matrix-array product time spent: %f s, and gpu memory bandwidth: %f GB/s.\n", time, bandwidth);
  cudaMemcpy(c, c_d, m*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFreeHost(sum);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(sum_d);
  cudaFree(ae_d);
  cudaFree(c_d);
}


int main()
{
  double *a, *b, *ccpu, *cgpu;
  int m = 1000;
  int n = 10000;
  a = (double*)aligned_alloc(sizeof(double), m*n*sizeof(double));
  b = (double*)aligned_alloc(sizeof(double),   n*sizeof(double));
  ccpu = (double*)aligned_alloc(sizeof(double),m*sizeof(double));
  cgpu = (double*)aligned_alloc(sizeof(double),m*sizeof(double));
  srand(time(NULL));
  int i;
  double checksum = 0.0;

  for(i = 0; i < m*n; ++i)
    a[i] = (double)rand()/RAND_MAX;
  for(i = 0; i < n; ++i)
    b[i] = (double)rand()/RAND_MAX;

  dotproduct_cpu(a, b, ccpu, m, n);

  dotproduct_gpu(a, b, cgpu, m, n);

  for(i = 0; i < m; ++i)
    checksum += fabs(ccpu[i] - cgpu[i]);

  printf("error checksum: %f\n", checksum);

  free(a);
  free(b);
  free(ccpu);
  free(cgpu);
  return 0;
}
