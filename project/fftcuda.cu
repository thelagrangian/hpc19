#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <time.h>
#include <cuda.h>

#define pi 3.14159265358979323846
#define BLOCK_SIZE 512
#define GRID_SIZE 8192

// g++ $(gsl-config --cflags) fft.cpp $(gsl-config --libs)


//void fft_gsl(double* x, int n)
//{
  /* x is a double array with length 2n,
 *  representing n complex numbers,
 *  where x[0,2,4,...,2n-2] are the real part of the n complex number,
 *  and   x[1,3,5,...,2n-1] are the imaginary part of the n complex numbers.
 */
//  gsl_fft_complex_radix2_backward(x, 1, n);
//}


const double c0 = 1.0;
const double c2 =-1.0/2.0;
const double c4 = 1.0/2.0/3.0/4.0;
const double c6 =-1.0/2.0/3.0/4.0/5.0/6.0;
const double c8 = 1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0;
const double c10=-1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0/9.0/10.0;
const double c1 = 1.0;
const double c3 =-1.0/2.0/3.0;
const double c5 = 1.0/2.0/3.0/4.0/5.0;
const double c7 =-1.0/2.0/3.0/4.0/5.0/6.0/7.0;
const double c9 = 1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0/9.0;
const double c11=-1.0/2.0/3.0/4.0/5.0/6.0/7.0/8.0/9.0/10.0/11.0;

struct complex_t
{ 
  double r;
  double i;
  complex_t(double r_=0.0, double i_=0.0):r(r_),i(i_){}
};

typedef struct complex_t complex_t;

double fcos(double);
double fsin(double);


double fcos(double x)
{ 
  if(x <= -0.75*pi)
    return -fcos(x + pi); 
  else if(-0.75*pi < x && x <= -0.25*pi)
    return fsin(x + 0.5*pi);
  else if(-0.25*pi < x && x <=  0.25*pi)
  { 
    double x0 = 1.0;
    double x2 = x*x;
    double x4 = x2*x2;
    double x6 = x4*x2;
    double x8 = x6*x2;
    double x10= x8*x2;
    return (x0*c0 + x2*c2 + x4*c4 + x6*c6 + x8*c8 + x10*c10);
  }
  else if( 0.25*pi < x && x <=  0.75*pi)
    return -fsin(x - 0.5*pi);
  else
    return -fcos(x - pi);
}

double fsin(double x)
{ 
  if(x <= -0.75*pi)
    return -fsin(x + pi); 
  else if(-0.75*pi < x && x <= -0.25*pi)
    return -fcos(x + 0.5*pi);
  else if(-0.25*pi < x && x <=  0.25*pi)
  { 
    double x1 = x;
    double x2 = x*x;
    double x3 = x1*x2;
    double x5 = x3*x2;
    double x7 = x5*x2;
    double x9 = x7*x2;
    double x11= x9*x2;
    return (x1*c1 + x3*c3 + x5*c5 + x7*c7 + x9*c9 + x11*c11);
  }
  else if( 0.25*pi < x && x <=  0.75*pi)
    return fcos(x - 0.5*pi);
  else
    return -fsin(x - pi);
}


// from http://www.katjaas.nl/bitreversal/bitreversal.html
__device__  __host__
unsigned long bitrev(unsigned long n, unsigned int bits)
{
    unsigned long nrev, N;
    unsigned int count;   
    N = 1UL<<bits;
    count = bits-1;   // initialize the count variable
    nrev = n;
    for(n>>=1; n; n>>=1)
    {
        nrev <<= 1;
        nrev |= n & 1;
        count--;
    }

    nrev <<= count;
    nrev &= N - 1;

    return nrev;
}

__global__
void bitrearrange_kernel(double*xr_d, double*xi_d, double*yr_d, double*yi_d, unsigned long n, unsigned int bits)
{
  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  while(tid < n)
  {
    yr_d[bitrev(tid, bits)] = xr_d[tid];
    yi_d[bitrev(tid, bits)] = xi_d[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__global__
void fftinner_kernel(double*yr_d, double*yi_d, double omega_m_r, double omega_m_i, unsigned long m, unsigned long n)
{
  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long k = tid * m;
  unsigned long j;
  while(k < n)
  {
    double omega_r = 1.0;
    double omega_i = 0.0;
    double t0_r, t0_i, t_r, t_i, u_r, u_i, t1_r, t1_i;
    for(j = 0; j < m/2; ++j)
    {
      t0_r = yr_d[k + j + m/2];
      t0_i = yi_d[k + j + m/2];
      t_r = omega_r*t0_r - omega_i*t0_i;
      t_i = omega_r*t0_i + omega_i*t0_r;
      u_r = yr_d[k + j];
      u_i = yi_d[k + j];
      yr_d[k + j] = u_r + t_r;
      yi_d[k + j] = u_i + t_i;
      yr_d[k + j + m/2] = u_r - t_r;
      yi_d[k + j + m/2] = u_i - t_i;

      t1_r = omega_r*omega_m_r - omega_i*omega_m_i;
      t1_i = omega_r*omega_m_i + omega_i*omega_m_r;
      omega_r = t1_r;
      omega_i = t1_i;
    }
    tid += blockDim.x*gridDim.x;
    k = tid * m;
  }

}

void ftt_cuda(double* xr, double*xi, double*yr, double*yi, unsigned long n)
{
  double*xr_d, *xi_d, *yr_d, *yi_d;
  //int i, m, k, j;
  unsigned long m;
  size_t memfree, memtotal;
  unsigned int bits = (unsigned int)log2((double)n);
  cudaMalloc(&xr_d, n*sizeof(double));
  cudaMalloc(&xi_d, n*sizeof(double));
  cudaMalloc(&yr_d, n*sizeof(double));
  cudaMalloc(&yi_d, n*sizeof(double));


  cudaMemcpy(xr_d, xr, n*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(xi_d, xi, n*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(yr_d, yr, n*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(yi_d, yi, n*sizeof(double), cudaMemcpyHostToDevice);


  bitrearrange_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(xr_d, xi_d, yr_d, yi_d, n, (unsigned int)log2((double)n));


  for(m = 2; m <= n; m*=2)
  {
    double theta = 2.0*pi/m;
    double omega_m_r = fcos(theta);
    double omega_m_i = fsin(theta);


    //int n2 = n/m;
    //int nblocks2 = (n2 + BLOCK_SIZE - 1)/BLOCK_SIZE;

    //(double*yr_d, double*yi_d, double omega_m_r, double omega_m_i, int m, int n)
    fftinner_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(yr_d, yi_d, omega_m_r, omega_m_i, m, n);
  }


  //cudaMemcpy(xr, xr_d, n*sizeof(double), cudaMemcpyDeviceToHost);
  //cudaMemcpy(xi, xi_d, n*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(yr, yr_d, n*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(yi, yi_d, n*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(xr_d);
  cudaFree(xi_d);
  cudaFree(yr_d);
  cudaFree(yi_d);

  cudaMemGetInfo(&memfree, &memtotal);
  //printf("free mem: %d, total mem: %d\n", memfree, memtotal);
}

void fft_seq(complex_t* x, complex_t*y, unsigned long n)
{
  unsigned long i, m, k, j;
  unsigned int bits = (int)log2((double)n);
  for(i = 0; i < n; ++i)
    y[bitrev(i, bits)] = x[i];


  for(m = 2; m <= n; m*=2)
  {
    double theta = 2.0*pi/m;
    complex_t omega_m(fcos(theta), fsin(theta));
    for(k = 0; k < n; k += m)
    {
      complex_t omega(1.0, 0.0);
      for(j = 0; j < m/2; ++j)
      {
        complex_t t0 = y[k + j + m/2];
        complex_t t(omega.r*t0.r - omega.i*t0.i, omega.r*t0.i + omega.i*t0.r);
        complex_t u = y[k + j];
        y[k + j] = complex_t(u.r + t.r, u.i + t.i);
        y[k + j + m/2] = complex_t(u.r - t.r, u.i - t.i);
        omega = complex_t(omega.r*omega_m.r - omega.i*omega_m.i, omega.r*omega_m.i + omega.i*omega_m.r);
      }
    }
  }
}

void fft_omp(complex_t* x, complex_t*y, int n)
{
  int i, m, k, j;
#pragma omp parallel
{
  int bits = (int)log2((double)n);

#pragma omp for
  for(i = 0; i < n; ++i)
    y[bitrev(i, bits)] = x[i];


  for(m = 2; m <= n; m*=2)
  {
    double theta = 2.0*pi/m;
    complex_t omega_m(fcos(theta), fsin(theta));
  #pragma omp for
    for(k = 0; k < n; k += m)
    {
      complex_t omega(1.0, 0.0);
      for(j = 0; j < m/2; ++j)
      {
        complex_t t0 = y[k + j + m/2];
        complex_t t(omega.r*t0.r - omega.i*t0.i, omega.r*t0.i + omega.i*t0.r);
        complex_t u = y[k + j];
        y[k + j] = complex_t(u.r + t.r, u.i + t.i);
        y[k + j + m/2] = complex_t(u.r - t.r, u.i - t.i);
        omega = complex_t(omega.r*omega_m.r - omega.i*omega_m.i, omega.r*omega_m.i + omega.i*omega_m.r);
      }
    }
  }

}

}


int main()
{
  unsigned long size_limit = (1UL<<30);
  unsigned long  n, i;
  struct timespec t0, t1;
  double timeint;
  srand(time(NULL));
  for(n = 8; n < size_limit; n *= 2)
  {
    //double    *x0 = (double*)malloc(2*n*sizeof(double));
    complex_t *x  = (complex_t*)malloc(n*sizeof(complex_t));
    complex_t *y  = (complex_t*)malloc(n*sizeof(complex_t));
    double    *xr = (double*)malloc(n*sizeof(double));
    double    *xi = (double*)malloc(n*sizeof(double));
    double    *yr = (double*)malloc(n*sizeof(double));
    double    *yi = (double*)malloc(n*sizeof(double));

    for(i = 0; i < n ; ++i)
    {
      xr[i] = x[i].r  = (double)rand()/RAND_MAX - 0.5;
      xi[i] = x[i].i  = (double)rand()/RAND_MAX - 0.5;
    }


    clock_gettime(CLOCK_MONOTONIC, &t0);
    ftt_cuda(xr, xi, yr, yi, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    timeint = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
    printf("problem size: %d, cuda: %f\n", n, timeint);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    fft_seq(x, y, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    timeint = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
    printf("problem size: %d, seq:  %f\n", n ,timeint);


    for(i = 0; i < n; ++i)
    {
      if(fabs(1.0 - yr[i]/y[i].r) > 1e-3)
        printf("%d, r: %8e, %8e\n", i, yr[i], y[i].r);
      if(fabs(1.0 - yi[i]/y[i].i) > 1e-3 )
        printf("%d, i: %8e, %8e\n", i, yi[i], y[i].i);
      //if(fabs(1.0 - yr[i]/xr[bitrev(i, (int)log2((double)n))]) > 1e-3)
      //  printf("%d, r: %8e, %8e\n", i, yr[i], y[i].r);
      //if(fabs(1.0 - yi[i]/xi[bitrev(i, (int)log2((double)n))]) > 1e-3 )
      //  printf("%d, i: %8e, %8e\n", i, yi[i], y[i].i);

    }


    free(x);
    free(y);
    free(xr);
    free(xi);
    free(yr);
    free(yi);
  }
  return 0;
}
