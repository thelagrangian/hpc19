#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<cuda.h>
#include<time.h>

#define BLOCK_SIZE (1<<10)

double dotproduct_omp(double*a, double*b, int n)
{
  struct timespec t0, t1;
  double time;
  double bandwidth;
  double rval = 0.0;
  int i;
  clock_gettime(CLOCK_MONOTONIC, &t0);
#pragma omp parallel for reduction(+:rval)
  for(i = 0; i < n; ++i)
    rval += a[i]*b[i];
  clock_gettime(CLOCK_MONOTONIC, &t1);
  time = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
  bandwidth = 2*sizeof(double)*n/1e9/time;
  printf("cpu dot product time spent: %f s, and cpu memory bandwidth: %f GB/s.\n", time, bandwidth);
  return rval;
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

double dotproduct_gpu(double*a, double*b, int n)
{
  struct timespec t0, t1;
  double time;
  double bandwidth;
  double *a_d, *b_d, *sum, *sum_d;
  int num_blocks = (n + BLOCK_SIZE - 1)/BLOCK_SIZE;
  sum = (double*)aligned_alloc(sizeof(double), num_blocks*sizeof(double));
  double rval = 0.0;

  cudaMalloc(&a_d, n*sizeof(double));
  cudaMalloc(&b_d, n*sizeof(double));
  cudaMalloc(&sum_d, num_blocks*sizeof(double));

  cudaMemcpy(a_d, a, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, n*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(sum_d, sum, num_blocks*sizeof(double), cudaMemcpyHostToDevice);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  dotproduct_kernel<<<num_blocks, BLOCK_SIZE>>>(sum_d, a_d, b_d, n);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  cudaMemcpy(sum, sum_d, num_blocks*sizeof(double), cudaMemcpyDeviceToHost);

  int i;
  for(i = 0; i < num_blocks; ++i)
    rval += sum[i];

  time = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
  bandwidth = 2*sizeof(double)*n/1e9/time;
  printf("gpu dot product time spent: %f s, and gpu memory bandwidth: %f GB/s.\n", time, bandwidth);

  free(sum);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(sum_d);

  return rval;
}


int main()
{
  double *a, *b;
  int n = 60000000;
  a = (double*)aligned_alloc(sizeof(double), n*sizeof(double));
  b = (double*)aligned_alloc(sizeof(double), n*sizeof(double));
  srand(time(NULL));
  int i;
  double dprod_cpu, dprod_gpu;
  for(i = 0; i < n; ++i)
  {
    a[i] = (double)rand()/RAND_MAX;
    b[i] = (double)rand()/RAND_MAX;
  }

  dprod_cpu = dotproduct_omp(a, b, n);

  dprod_gpu = dotproduct_gpu(a, b, n);
  //printf("cpu dot product: %f\ngpu dot product: %f\n", dprod_cpu, dprod_gpu);
  printf("error checksum: %f\n", fabs(dprod_cpu - dprod_gpu));

  free(a);
  free(b);
  return 0;
}
