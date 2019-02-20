#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

double timelapse(struct timespec*start, struct timespec*end)
{
  return (end->tv_sec + end->tv_nsec/1.0e9)-(start->tv_sec + start->tv_nsec/1.0e9);
}

void jacobi_seq(int n, double h, double*u, double reltol, int maxiter)
{
  int iter = 0;
  double* x = u;
  double* y = (double*)malloc((n+2)*(n+2)*sizeof(double));
  int i, j;
  double err;
  double tol;
  while(iter<maxiter)
  {
    err = 0.0;
    ++iter;
    // point-wise calculation
    for(i = 1; i < n + 1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        if(iter%2 == 1)
          y[i*n+j] = 0.25*(h*h + x[(i-1)*n+j] + x[i*n+j-1] + x[(i+1)*n+j] + x[i*n+j+1]);
        else
          x[i*n+j] = 0.25*(h*h + y[(i-1)*n+j] + y[i*n+j-1] + y[(i+1)*n+j] + y[i*n+j+1]);
      }
    }

    // error calculation
    double diff;
    for(i = 1; i < n +1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        if(iter%2==1)
          diff = (4*y[i*n+j] - y[(i-1)*n+j] -y[i*n+j-1] -y[(i+1)*n+j] -y[i*n+j+1])/h/h - 1;
        else
          diff = (4*x[i*n+j] - x[(i-1)*n+j] -x[i*n+j-1] -x[(i+1)*n+j] -x[i*n+j+1])/h/h - 1;
        err = diff*diff;
      }
    }
    err = sqrt(err);

    if(iter==1)
      tol = reltol*err;
    if(err < tol)
      break;
  }
  printf("jacobi. original error: %f, final error: %f, iteration: %d\n", tol/reltol, err, iter);
  if(iter%2==1)
  {
    for(i = 1; i < n + 1; ++i)
      for(j = 1; j < n + 1; ++j)
        u[i*n+j] = y[i*n+j];
  }

  free(y);
}


void gauss_seidel_seq(int n, double h, double*u, double reltol, int maxiter)
{
  int iter = 0;
  int i, j;
  double err;
  double tol;
  while(iter<maxiter)
  {
    ++iter;
    err = 0.0;

    // update red points
    for(i = 1; i < n + 1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        if((i+j)%2==0)
          u[i*n+j] = 0.25*(h*h + u[(i-1)*n+j] + u[i*n+j-1] + u[(i+1)*n+j] + u[i*n+j+1]);
      }
    }

    // update black points
    for(i = 1; i < n + 1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        if((i+j)%2==1)
          u[i*n+j] = 0.25*(h*h + u[(i-1)*n+j] + u[i*n+j-1] + u[(i+1)*n+j] + u[i*n+j+1]);
      }
    }

    double diff;
    for(i = 1; i < n +1; ++i)
    {
      for(j = 1; j < n + 1; ++j)
      {
        diff = (4*u[i*n+j] - u[(i-1)*n+j] - u[i*n+j-1] - u[(i+1)*n+j] - u[i*n+j+1])/h/h - 1;
        err = diff*diff;
      }
    }
    err = sqrt(err);

    if(iter==1)
      tol = reltol*err;
    if(err < tol)
      break;
  }


  printf("gauss-seidel. original error: %f, final error: %f, iteration: %d\n", tol/reltol, err, iter);
}

int main()
{
  int n = 4;
  int iter = 10;
  int i, j;
  struct timespec start, end;
  double timespent;
  for(i = 0; i < iter; ++i)
  {
    n *= 2;
    double*u = (double*)malloc((n+2)*(n+2)*sizeof(double));
    double*v = (double*)malloc((n+2)*(n+2)*sizeof(double));
    for(j = 0; j < (n+2)*(n+2); ++j)
    {
      u[j] = 0;
      v[j] = 0;
    }
    double h = 1.0/(n+1);
    clock_gettime(CLOCK_MONOTONIC, &start);
    jacobi_seq(n, h, u, 1.0e-6, 40000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("jacobi. n: %d, time: %f\n", n, timespent);


    clock_gettime(CLOCK_MONOTONIC, &start);
    gauss_seidel_seq(n, h, v, 1.0e-6, 40000);
    clock_gettime(CLOCK_MONOTONIC, &end);
    timespent = timelapse(&start, &end);
    printf("gauss-seidel. n: %d, time: %f\n", n, timespent);

    free(u);
    free(v);
  }
  
  return 0;
}
