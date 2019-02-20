#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

int jacobi(int n, double h, double *u, double *f, double tolrate, int iter, double*errj)
{
  double *u0 = (double*)malloc(n*sizeof(double));
  double *u1 = (double*)malloc(n*sizeof(double));
  double *res= (double*)malloc(n*sizeof(double));
  double err, tol;
  int i, j, counter=0;
  for(i=0; i<n; ++i)
    u0[i] = u[i];
  while(counter<iter)
  {
    if(counter%2==0)
    {
      for(i=0; i<n; ++i)
      {
        res[i] = -f[i];
        for(j = i-1; j<i+2; ++j)
        {
          if(j>=0 && j<n)
          {
            if(i==j)
              res[i] += 2.0/h/h*u0[j];
            else
              res[i] += (-1.0)/h/h*u0[j];
          }
        }
      }
    }
    else
    {
      for(i=0; i<n; ++i)
      {
        res[i] = -f[i];
        for(j=i-1; j<i+2; ++j)
        {
          if(j>=0 && j<n)
          {
            if(i==j)
              res[i] += 2.0/h/h*u1[j];
            else
              res[i] += (-1.0)/h/h*u1[j];
          }
        }
      }
    }
    err=0;
    for(i=0; i<n; ++i)
      err += res[i]*res[i];
    err = sqrt(err);
    //for(i=0;i<n;++i)
    //  err += abs(res[i]);
    if(counter==0)
      tol = tolrate*err;
    errj[counter] = err;
    if(err<tol)
      break;
    if(counter%2==0)
    {
      for(i=0;i<n;++i)
      {
        u1[i] = f[i];
        for(j=i-1;j<i+2;++j)
        {
          if(j>=0 && j<n)
          {
            if(j!=i)
              u1[i] -= (-1.0)/h/h*u0[j];
          }
        }
        u1[i] /= (2.0/h/h);
      }
    }
    else
    {
      for(i=0;i<n;++i)
      { 
        u0[i] = f[i];
        for(j=i-1;j<i+2;++j)
        {
          if(j>=0 && j<n)
          {
            if(j!=i)
              u0[i] -= (-1.0/h/h)*u1[j];
          }
        }
        u0[i] /= (2.0/h/h);
      }
    }
    ++counter;
  }
  if(counter%2==0)
  {
    for(i=0;i<n;++i)
      u[i] = u0[i];
  }
  else
  {
    for(i=0;i<n;++i)
      u[i] = u1[i];
  }
  free(u0);
  free(u1);
  free(res);
  return counter;
}

int gauss_seidel(int n, double h, double *u, double *f, double tolrate, int iter, double*errgs)
{
  int counter = 0;
  double *res = (double*)malloc(n*sizeof(double));
  double tol, err;
  int i,j;
  while(counter<iter)
  {
    for(i=0;i<n;++i)
    {
      res[i] = -f[i];
      for(j=i-1;j<i+2;++j)
      {
        if(j>=0 && j<n)
        {
          if(i==j)
            res[i] += 2.0/h/h*u[j];
          else
            res[i] += (-1.0)/h/h*u[j];
        }
      }
    }
    err = 0.0;
    for(i=0;i<n;++i)
      err += res[i]*res[i];
    err = sqrt(err);
    //for(i=0;i<n;++i)
    //  err += abs(res[i]);
    if(counter==0)
      tol = tolrate*err;
    errgs[counter] = err;
    if(err<tol)
      break;
    for(i=0;i<n;++i)
    {
      u[i] = f[i];
      for(j=i-1;j<i+2;++j)
      {
        if(j>=0 && j<n)
        {
          if(j!=i)
            u[i] -= (-1.0)/h/h*u[j];
        }
      }
      u[i] /= (2.0/h/h);
    }

    ++counter;
  }
  free(res);
  return counter;
}

int main(int argc, int* argv[])
{
  if(argc<2)
  {
    printf("Discretization parameter N needs to be provided.\n");
    printf("Program aborted.\n");
    exit(EXIT_FAILURE);
  }
  int n = atoi(argv[1]);
  double h  = 1.0/(double)(n+1);
  double *uj = (double*)calloc(n, sizeof(double));
  double *ugs= (double*)calloc(n, sizeof(double));
  double *f = (double*)malloc(n*sizeof(double));
  double tolrate = 1.0/1e6;
  int iter = 500000;
  double *errj = (double*)malloc(iter*sizeof(double));
  double*errgs = (double*)malloc(iter*sizeof(double));
  int i,j;
  for(i=0; i<n; ++i)
    f[i] = 1.0;

  for(i=0;i<iter;++i)
  {
    errj[i] = -1.0;
    errgs[i]= -1.0;
  }
  int jacobiiter, gsiter;
  struct timespec tj0, tj1, tgs0, tgs1;


  clock_gettime(CLOCK_MONOTONIC, &tj0);
  jacobiiter = jacobi(n, h, uj, f, tolrate, iter, errj);
  clock_gettime(CLOCK_MONOTONIC, &tj1);


  clock_gettime(CLOCK_MONOTONIC, &tgs0);
  gsiter     = gauss_seidel(n, h, ugs, f, tolrate, iter, errgs);
  clock_gettime(CLOCK_MONOTONIC, &tgs1);

  double timej, timegs;
  timej = (tj1.tv_sec + tj1.tv_nsec/1.0e9)-(tj0.tv_sec+ tj0.tv_nsec/1.0e9);
  timegs= (tgs1.tv_sec + tgs1.tv_nsec/1.0e9)-(tgs0.tv_sec + tgs0.tv_nsec/1.0e9);
  printf("Problem dimension: %d.\n", n);
  printf("Jacobi L2-norm error series:\n");
  for(i=0;i<iter;++i)
    if(errj[i]!=-1)
      printf("%d-th: %f\n",i ,errj[i]);
  printf("Gauss-Seidel L2-norm error series:\n");
  for(i=0;i<iter;++i)
    if(errgs[i]!=-1)
      printf("%d-th: %f\n",i, errgs[i]);
  printf("Jacobi solution:\n");
  for(i=0;i<n;++i)
    printf("%d %f\n",i,uj[i]);
  printf("Gauss-Seidel solution:\n");
  for(i=0;i<n;++i)
    printf("%d %f\n",i,ugs[i]);
  printf("Jacobi: %d iterations, %f seconds.\n", jacobiiter, timej);
  printf("Gauss-Seidel: %d iterations, %f seconds.\n", gsiter, timegs);
  free(uj);
  free(ugs);
  free(f);
  free(errj);
  free(errgs);
  return 0;
}
