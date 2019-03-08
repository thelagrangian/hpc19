#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>


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
  printf("gs omp, size: %d, iteration: %d\n", n, iter);

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
    gs_seq(n, h, u, 1.0e-6, 40000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("gs seq. n: %d, time: %f\n", n, timespent);


    clock_gettime(CLOCK_MONOTONIC, &start);
    gs_omp(n, h, v, 1.0e-6, 40000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("gs omp. n: %d, time: %f\n", n, timespent);


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
      gs_omp(n,h,u,1.0e-12,4000,nthreads);
      clock_gettime(CLOCK_MONOTONIC, &end);
      timespent = timelapse(&start, &end);
      printf("gs omp, n: %d, nthreads: %d, time: %f\n", n, nthreads, timespent);

      free(u);
    }
#else
    double*u = (double*)calloc((n+2)*(n+2), sizeof(double));
    double h = 1.0/(n+1);

    clock_gettime(CLOCK_MONOTONIC, &start);
    gs_seq(n,h,u,1.0e-15,4000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("gs seq, n: %d, time: %f\n", n, timespent);

    free(u);
#endif
  }
  
  return 0;
}
