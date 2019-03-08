#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>


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

    if(iter==1)
      tol = reltol*err;
    if(err < tol)
      break;

  }

  //printf("jacobi seq %d, original error: %f, final error: %f, iteration: %d\n", n, tol/reltol, err, iter);
  printf("jacobi omp, size: %d, iteration: %d\n", n, iter);

  if(iter%2==1)
    memcpy(u, y, sizeof(double)*(n+2)*(n+2));

  free(y);

}

int main()
{
  int n = 16;
  int iter = 6;
  int nthreads;
  int titer = 5;
  int i, j;
  struct timespec start, end;
  double timespent;
  /* Original testing
  for(i = 0; i < iter; ++i)
  {
    n *= 2;
    double*u = (double*)calloc((n+2)*(n+2), sizeof(double));
    double*v = (double*)calloc((n+2)*(n+2), sizeof(double));


    double h = 1.0/(n+1);


    clock_gettime(CLOCK_MONOTONIC, &start);
    jacobi_seq(n, h, u, 1.0e-6, 40000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("jacobi seq. n: %d, time: %f\n", n, timespent);


    clock_gettime(CLOCK_MONOTONIC, &start);
    jacobi_omp(n, h, v, 1.0e-6, 40000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("jacobi omp. n: %d, time: %f\n", n, timespent);


    free(u);
    free(v);
  }
  */
  for(i = 0; i<iter;++i)
  {
    n *= 2;
#ifdef _OPENMP
    nthreads = 2;
    for(j=0;j<titer;++j)
    {
      nthreads *=2;

      double*u = (double*)calloc((n+2)*(n+2), sizeof(double));
      double h = 1.0/(n+1);

      clock_gettime(CLOCK_MONOTONIC, &start);
      jacobi_omp(n,h,u,1.0e-12,4000,nthreads);
      clock_gettime(CLOCK_MONOTONIC, &end);
      timespent = timelapse(&start, &end);
      printf("jacobi omp, n: %d, nthreads: %d, time: %f\n", n, nthreads, timespent);

      free(u);
    }
#else
    double*u = (double*)calloc((n+2)*(n+2), sizeof(double));
    double h = 1.0/(n+1);

    clock_gettime(CLOCK_MONOTONIC, &start);
    jacobi_seq(n,h,u,1.0e-12,4000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("jacobi seq, n: %d, time: %f\n", n, timespent);

    free(u);
#endif
  }
  
  
  return 0;
}
