#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>

#define BLOCK_SIZE 32

double timelapse(struct timespec*start, struct timespec*end)
{
  return (end->tv_sec + end->tv_nsec/1.0e9)-(start->tv_sec + start->tv_nsec/1.0e9);
}


void gs_seq(int n, double h, double*u, double reltol, int maxiter)
{
  int iter = 0;
  int i, j;
  double err;
  double tol;

  while(iter<maxiter)
  {
    ++iter;
    err = 0.0;


    // update red points (i+j)%2==0
    for(i = 1; i < n + 1; ++i)
    {
      for(j = i; j < n + 1; j += 2)
      {
        int idx = i*(n+2)+j;
        u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }

    for(i = 1; i < n + 1; ++i)
    {
      for(j = i-2; j >=1 ; j -= 2)
      {
        int idx = i*(n+2)+j;
        u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }

    // update black points (i+j)%2==1
    for(i = 1; i < n + 1; ++i)
    {
      for(j = i+1; j < n + 1; j +=2)
      {
        int idx = i*(n+2)+j;
        u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }


    for(i = 1; i < n + 1; ++i)
    {
      for(j = i-1; j >=1 ; j -=2)
      {
        int idx = i*(n+2)+j;
        u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }


/*
    for(i = 1; i < n + 1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        int idx = i*(n+2)+j;
        if((i+j)%2==0)
          u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }


    for(i = 1; i < n + 1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        int idx = i*(n+2)+j;
        if((i+j)%2==1)
          u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }

*/

    // calculate err residual
    for(i = 1; i < n +1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        int idx = i*(n+2)+j;
        double diff = (4*u[idx] - u[idx+1] - u[idx-1] - u[idx+n+2] - u[idx-n-2])/h/h - 1;
        err = err + diff*diff;
      }
    }
    err = sqrt(err);

    if(iter==1)
      tol = reltol*err;
    if(err < tol)
      break;
  }

  //printf("gs seq %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("gs seq, size: %d, iteration: %d\n", n, iter);
}


void gs_omp(int n, double h, double*u, double reltol, int maxiter, int nthreads)
{
  int iter = 0;
  int i, j;
  double err;
  double tol;

  while(iter<maxiter)
  {
    ++iter;
    err = 0.0;

#pragma omp parallel num_threads(nthreads)
{

    // update red points (i+j)%2==0
  #pragma omp for collapse(2)
    for(i = 1; i < n + 1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        int idx = i*(n+2)+j;
        if((i+j)%2==0)
          u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }


    // update black points (i+j)%2==1
  #pragma omp for collapse(2)
    for(i = 1; i < n + 1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        int idx = i*(n+2)+j;
        if((i+j)%2==1)
          u[idx] = 0.25*(h*h + u[idx+1] + u[idx-1] + u[idx+n+2] + u[idx-n-2]);
      }
    }

    // calculate err residual
  #pragma omp for collapse(2) reduction(+:err)
    for(i = 1; i < n +1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        int idx = i*(n+2)+j;
        double diff = (4*u[idx] - u[idx+1] - u[idx-1] - u[idx+n+2] - u[idx-n-2])/h/h - 1;
        err = err + diff*diff;
      }
    }

}

    err = sqrt(err);

    if(iter==1)
      tol = reltol*err;
    if(err < tol)
      break;
  }

  //printf("gs omp %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("gs omp, size: %d, iteration: %d,", n, iter);

}

__global__ void gs_kernel0(double*u_d, int n, double h, double *err_d, int num_blocks)
{
  __shared__ double su[BLOCK_SIZE][BLOCK_SIZE];
  int gidxx = threadIdx.x + blockIdx.x * (BLOCK_SIZE - 2);
  int gidxy = threadIdx.y + blockIdx.y * (BLOCK_SIZE - 2);

  if(gidxx < n+2 && gidxy < n+2)
    su[threadIdx.x][threadIdx.y] = u_d[gidxx*(n+2) + gidxy];
  __syncthreads();

  if(threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.x != BLOCK_SIZE -1 && threadIdx.y != BLOCK_SIZE -1 && gidxx < n+1 && gidxy < n+1 && (gidxx + gidxy)%2==0)
    u_d[gidxx*(n+2) + gidxy] = 0.25*(h*h + su[threadIdx.x+1][threadIdx.y] + su[threadIdx.x-1][threadIdx.y] + su[threadIdx.x][threadIdx.y+1] + su[threadIdx.x][threadIdx.y-1]);
}

__global__ void gs_kernel1(double*u_d, int n, double h, double *err_d, int num_blocks)
{
  __shared__ double su[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double serr[BLOCK_SIZE*BLOCK_SIZE];
  int gidxx = threadIdx.x + blockIdx.x * (BLOCK_SIZE - 2);
  int gidxy = threadIdx.y + blockIdx.y * (BLOCK_SIZE - 2);

  serr[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0.0;

  if(gidxx < n+2 && gidxy < n+2)
    su[threadIdx.x][threadIdx.y] = u_d[gidxx*(n+2) + gidxy];
  __syncthreads();

  if(threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.x != BLOCK_SIZE -1 && threadIdx.y != BLOCK_SIZE -1 && gidxx < n+1 && gidxy < n+1 && (gidxx + gidxy)%2==1)
    u_d[gidxx*(n+2) + gidxy] = 0.25*(h*h + su[threadIdx.x+1][threadIdx.y] + su[threadIdx.x-1][threadIdx.y] + su[threadIdx.x][threadIdx.y+1] + su[threadIdx.x][threadIdx.y-1]);
  __syncthreads();

  if(gidxx < n+2 && gidxy < n+2)
    su[threadIdx.x][threadIdx.y] = u_d[gidxx*(n+2) + gidxy];
  __syncthreads();

  double diff;
  if(threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.x != BLOCK_SIZE -1 && threadIdx.y != BLOCK_SIZE -1 && gidxx < n+1 && gidxy < n+1)
  {
    diff = (4.0*su[threadIdx.x][threadIdx.y] - su[threadIdx.x+1][threadIdx.y] - su[threadIdx.x-1][threadIdx.y] - su[threadIdx.x][threadIdx.y+1] - su[threadIdx.x][threadIdx.y-1])/h/h - 1.0;
    serr[threadIdx.x * BLOCK_SIZE + threadIdx.y] = diff*diff;
  }
  __syncthreads();

  int stride = BLOCK_SIZE*BLOCK_SIZE;

  while(stride > 1)
  {
    stride /= 2;
    if(threadIdx.x * BLOCK_SIZE + threadIdx.y < stride)
      serr[threadIdx.x * BLOCK_SIZE + threadIdx.y] += serr[threadIdx.x * BLOCK_SIZE + threadIdx.y + stride];
    __syncthreads();
  }

  if(threadIdx.x == 0 && threadIdx.y ==0)
    err_d[num_blocks * blockIdx.x + blockIdx.y] = serr[0];
}

void gs_cuda(int n, double h, double*u, double reltol, int maxiter)
{
  int iter = 0;
  int num_blocks = (n + BLOCK_SIZE - 3)/(BLOCK_SIZE - 2);

  double * u_d, *err_d, *err;
  cudaMalloc(&u_d, (n+2)*(n+2)*sizeof(double));
  cudaMalloc(&err_d, num_blocks*num_blocks*sizeof(double));
  cudaMallocHost(&err, num_blocks*num_blocks*sizeof(double));

  cudaMemcpy(u_d, u, (n+2)*(n+2)*sizeof(double), cudaMemcpyHostToDevice);

  int i;
  double error;
  double tol;

  dim3 blockD(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridD(num_blocks, num_blocks);

  while(iter<maxiter)
  {
    ++iter;
    error = 0.0;

    gs_kernel0<<<gridD, blockD>>>(u_d, n, h, err_d, num_blocks);
    gs_kernel1<<<gridD, blockD>>>(u_d, n, h, err_d, num_blocks);
    cudaMemcpy(err, err_d, num_blocks*num_blocks*sizeof(double), cudaMemcpyDeviceToHost);

#pragma omp parallel for reduction(+:error)
    for(i=0; i < num_blocks*num_blocks; ++i)
      error += err[i];

    error = sqrt(error);

    if(iter==1)
      tol = reltol*error;
    if(error < tol)
      break;
  }

  //printf("gs omp %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("gs cuda, size: %d, iteration: %d,", n, iter);

  cudaMemcpy(u, u_d, (n+2)*(n+2)*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(u_d);
  cudaFree(err_d);
  cudaFreeHost(err);
}

int main()
{
  int n = 4;//16;
  int iter = 6;
  int nthreads;
  int titer = 5;
  int i, j;
  struct timespec start, end;
  double timespent;

  for(i = 0; i<iter;++i)
  {
    n *= 2;
#ifdef _OPENMP
    nthreads = 2;
    double h = 1.0/(n+1);
    for(j=0;j<titer;++j)
    {
      nthreads *=2;

      double*u = (double*)calloc((n+2)*(n+2), sizeof(double));
      clock_gettime(CLOCK_MONOTONIC, &start);
      gs_omp(n,h,u,1.0e-12,10000,nthreads);
      clock_gettime(CLOCK_MONOTONIC, &end);
      timespent = timelapse(&start, &end);
      printf(" time: %f\n", timespent);
      free(u);
    }

    double*v = (double*)calloc((n+2)*(n+2), sizeof(double));
    clock_gettime(CLOCK_MONOTONIC, &start);
    gs_cuda(n,h,v,1.0e-12,10000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf(" time: %f\n", timespent);
    free(v);
#else
    double*u = (double*)calloc((n+2)*(n+2), sizeof(double));
    double*v = (double*)calloc((n+2)*(n+2), sizeof(double));
    double h = 1.0/(n+1);

    clock_gettime(CLOCK_MONOTONIC, &start);
    jacobi_seq(n,h,u,1.0e-12,10000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf(" time: %f\n", timespent);

    clock_gettime(CLOCK_MONOTONIC, &start);
    gs_cuda(n,h,u,1.0e-12,10000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf(" time: %f\n", timespent);

    free(v);
    free(u);

#endif
  }
  
  return 0;
}
