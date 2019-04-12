#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<float.h>

#define BLOCK_SIZE 32

double timelapse(struct timespec*start, struct timespec*end)
{
  return (end->tv_sec + end->tv_nsec/1.0e9)-(start->tv_sec + start->tv_nsec/1.0e9);
}


void jacobi_seq(int n, double h, double*u, double reltol, int maxiter)
{
  int iter = 0;
  double* x = u;
  double* y = (double*)calloc((n+2)*(n+2), sizeof(double));
  int i, j;
  double err;
  double tol;

  while(iter<maxiter)
  {
    err = 0.0;
    ++iter;

    // point-wise calculation
    if(iter%2 == 1)
    {
      for(i = 1; i < n + 1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          int idx = i*(n+2) + j;
          y[idx] = 0.25*(h*h + x[idx+1] + x[idx-1] + x[idx+n+2] + x[idx-n-2]);
        }
      }
    }
    else
    {
      for(i = 1; i < n + 1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          int idx = i*(n+2) + j;
          x[idx] = 0.25*(h*h + y[idx+1] + y[idx-1] + y[idx+n+2] + y[idx-n-2]);
        }
      } 
    }


    // error calculation
    if(iter%2==1)
    {
      for(i = 1; i < n +1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          double diff;
          int idx = i*(n+2) + j;
          diff = (4*y[idx] - y[idx+1] -y[idx-1] -y[idx+n+2] -y[idx-n-2])/h/h - 1;
          err = err + diff*diff;
        }
      } 
    }
    else
    {
      for(i = 1; i < n +1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          double diff;
          int idx = i*(n+2) + j;
          diff = (4*x[idx] - x[idx+1] -x[idx-1] -x[idx+n+2] -x[idx-n-2])/h/h - 1;
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

  //printf("jacobi seq %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("jacobi seq, size: %d, iteration: %d\n", n, iter);

  if(iter%2==1)
    memcpy(u, y, sizeof(double)*(n+2)*(n+2));

  free(y);

}

void jacobi_omp2(int n, double h, double*u, double reltol, int maxiter, int nthreads)
{
  int iter = 0;
  double* x = u;
  double* y = (double*)calloc((n+2)*(n+2), sizeof(double));
  int i, j;
  double err= 0.0;
  double tol;


#pragma omp parallel num_threads(nthreads)
{

  while(1)
  {

  #pragma omp single
  {
    err = 0.0;
  }
  //#pragma omp barrier


    // point-wise calculation
    if(iter%2 == 1)
    {
  #pragma omp for collapse(2)
      for(i = 1; i < n + 1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          int idx = i*(n+2) + j;
          y[idx] = 0.25*(h*h + x[idx+1] + x[idx-1] + x[idx+n+2] + x[idx-n-2]);
        }
      }
    }
    else
    {
  #pragma omp for collapse(2)
      for(i = 1; i < n + 1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          int idx = i*(n+2) + j;
          x[idx] = 0.25*(h*h + y[idx+1] + y[idx-1] + y[idx+n+2] + y[idx-n-2]);
        }
      } 
    }


    // error calculation
    if(iter%2==1)
    {
  #pragma omp for collapse(2) reduction (+:err)
      for(i = 1; i < n +1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          double diff;
          int idx = i*(n+2) + j;
          diff = (4*y[idx] - y[idx+1] -y[idx-1] -y[idx+n+2] -y[idx-n-2])/h/h - 1;
          err = err + diff*diff;
        }
      } 
    }
    else
    {
  #pragma omp for collapse(2) reduction (+:err)
      for(i = 1; i < n +1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          double diff;
          int idx = i*(n+2) + j;
          diff = (4*x[idx] - x[idx+1] -x[idx-1] -x[idx+n+2] -x[idx-n-2])/h/h - 1;
          err = err + diff*diff;
        }
      }
    }

  #pragma omp single
  {
    err = sqrt(err);
    ++iter;
    if(iter==1)
      tol = reltol*err;
  }
  if(iter>=maxiter || err < tol)
    break;
  #pragma omp barrier


  }

}


  //printf("jacobi seq %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("jacobi omp, size: %d, iteration: %d\n", n, iter);

  if(iter%2==1)
    memcpy(u, y, sizeof(double)*(n+2)*(n+2));

  free(y);

}



void jacobi_omp(int n, double h, double*u, double reltol, int maxiter, int nthreads)
{
  int iter = 0;
  double* x = u;
  double* y = (double*)calloc((n+2)*(n+2), sizeof(double));
  int i, j;
  double err;
  double tol;

  while(iter<maxiter)
  {
    err = 0.0;
    ++iter;


#pragma omp parallel num_threads(nthreads)
{

    // point-wise calculation
    if(iter%2 == 1)
    {
  #pragma omp for collapse(2)
      for(i = 1; i < n + 1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          int idx = i*(n+2) + j;
          y[idx] = 0.25*(h*h + x[idx+1] + x[idx-1] + x[idx+n+2] + x[idx-n-2]);
        }
      }
    }
    else
    {
  #pragma omp for collapse(2)
      for(i = 1; i < n + 1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          int idx = i*(n+2) + j;
          x[idx] = 0.25*(h*h + y[idx+1] + y[idx-1] + y[idx+n+2] + y[idx-n-2]);
        }
      } 
    }


    // error calculation
    if(iter%2==1)
    {
  #pragma omp for collapse(2) reduction (+:err)
      for(i = 1; i < n +1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          double diff;
          int idx = i*(n+2) + j;
          diff = (4*y[idx] - y[idx+1] -y[idx-1] -y[idx+n+2] -y[idx-n-2])/h/h - 1;
          err = err + diff*diff;
        }
      } 
    }
    else
    {
  #pragma omp for collapse(2) reduction (+:err)
      for(i = 1; i < n +1; ++i)
      {
        for(j = 1; j < n + 1; ++j)
        {
          double diff;
          int idx = i*(n+2) + j;
          diff = (4*x[idx] - x[idx+1] -x[idx-1] -x[idx+n+2] -x[idx-n-2])/h/h - 1;
          err = err + diff*diff;
        }
      }
    }

}

    err = sqrt(err);
    //printf("%d: %f\n", iter, err);

    if(iter==1)
      tol = reltol*err;
    if(err < tol)
      break;

  }

  //printf("jacobi seq %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("jacobi omp, size: %d, iteration: %d,", n, iter);

  if(iter%2==1)
    memcpy(u, y, sizeof(double)*(n+2)*(n+2));

  free(y);

}


__global__ void jacobi_kernel(double*y_d, double*x_d, int n, double h, double*err_d, int num_blocks)
{
  __shared__ double sx[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double sy[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double serr[BLOCK_SIZE*BLOCK_SIZE];
  int gidxx = threadIdx.x + blockIdx.x * (BLOCK_SIZE - 2);
  int gidxy = threadIdx.y + blockIdx.y * (BLOCK_SIZE - 2);
  double diff;

  serr[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0.0;

  if(gidxx < n+2 && gidxy < n+2)
  {
    sx[threadIdx.x][threadIdx.y] = x_d[gidxx*(n+2) + gidxy];
    sy[threadIdx.x][threadIdx.y] = x_d[gidxx*(n+2) + gidxy];
  }
  __syncthreads();

  if(threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.x != BLOCK_SIZE -1 && threadIdx.y != BLOCK_SIZE -1 && gidxx < n+1 && gidxy < n+1)
  {
    sy[threadIdx.x][threadIdx.y] = 0.25*(h*h + sx[threadIdx.x+1][threadIdx.y] + sx[threadIdx.x-1][threadIdx.y] + sx[threadIdx.x][threadIdx.y+1] + sx[threadIdx.x][threadIdx.y-1]);
    y_d[gidxx*(n+2) + gidxy] = sy[threadIdx.x][threadIdx.y];
  }
  __syncthreads();

  //if(gidxx < n+2 && gidxy < n+2)
  //  y_d[gidxx*(n+2) + gidxy] = sy[threadIdx.x][threadIdx.y];
  //__syncthreads();

  //diff = (4*y[idx] - y[idx+1] -y[idx-1] -y[idx+n+2] -y[idx-n-2])/h/h - 1;
  if(threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.x != BLOCK_SIZE -1 && threadIdx.y != BLOCK_SIZE -1 && gidxx < n+1 && gidxy < n+1)
  {
    diff = (4.0*sy[threadIdx.x][threadIdx.y] - sy[threadIdx.x+1][threadIdx.y] - sy[threadIdx.x-1][threadIdx.y] - sy[threadIdx.x][threadIdx.y+1] - sy[threadIdx.x][threadIdx.y-1])/h/h - 1.0;
    //diff = (4*y_d[gidxx*(n+2) + gidxy] - y_d[(gidxx-1)*(n+2) + gidxy] -y_d[(gidxx+1)*(n+2) + gidxy] -y_d[gidxx*(n+2) + gidxy-1] -y_d[gidxx*(n+2) + gidxy+1])/h/h - 1;
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


void jacobi_cuda(int n, double h, double*u, double reltol, int maxiter)
{
  int iter = 0;
  int num_blocks = (n + BLOCK_SIZE - 3)/(BLOCK_SIZE - 2);

  double *x_d, *y_d, *err_d, *err;
  cudaMalloc(&x_d, (n+2)*(n+2)*sizeof(double));
  cudaMalloc(&y_d, (n+2)*(n+2)*sizeof(double));
  cudaMalloc(&err_d, num_blocks*num_blocks*sizeof(double));
  cudaMallocHost(&err, num_blocks*num_blocks*sizeof(double));

  cudaMemcpy(x_d, u, (n+2)*(n+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, u, (n+2)*(n+2)*sizeof(double), cudaMemcpyHostToDevice);

  int i;
  double error;
  double tol;

  dim3 blockD(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridD(num_blocks, num_blocks);

  while(iter<maxiter)
  {
    error = 0.0;
    ++iter;

    if(iter%2==1)
      jacobi_kernel<<<gridD, blockD>>>(y_d, x_d, n, h, err_d, num_blocks);
    else
      jacobi_kernel<<<gridD, blockD>>>(x_d, y_d, n, h, err_d, num_blocks);
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

  //printf("jacobi seq %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("jacobi cuda, size: %d, iteration: %d,", n, iter);

  if(iter%2==1)
    cudaMemcpy(u, y_d, (n+2)*(n+2)*sizeof(double), cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(u, x_d, (n+2)*(n+2)*sizeof(double), cudaMemcpyDeviceToHost);
    //memcpy(u, y, sizeof(double)*(n+2)*(n+2));

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(err_d);
  cudaFreeHost(err);

}

int main()
{
  int n = 4;//16;
  int iter = 7;
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
      jacobi_omp(n,h,u,1.0e-12,10000,nthreads);
      clock_gettime(CLOCK_MONOTONIC, &end);
      timespent = timelapse(&start, &end);
      printf(" time: %f, no threads: %d\n", timespent, nthreads);
      free(u);
    }

    double*v = (double*)calloc((n+2)*(n+2), sizeof(double));
    clock_gettime(CLOCK_MONOTONIC, &start);
    jacobi_cuda(n,h,v,1.0e-12,10000);
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
    jacobi_cuda(n,h,u,1.0e-12,10000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf(" time: %f\n", timespent);

    free(v);
    free(u);
#endif
  }
  
  
  return 0;
}
