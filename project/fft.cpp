#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <time.h>

// g++ $(gsl-config --cflags) fft.cpp $(gsl-config --libs)


void fft_gsl(double* x, int n)
{
  /* x is a double array with length 2n,
 *  representing n complex numbers,
 *  where x[0,2,4,...,2n-2] are the real part of the n complex number,
 *  and   x[1,3,5,...,2n-1] are the imaginary part of the n complex numbers.
 */
  gsl_fft_complex_radix2_backward(x, 1, n);
}


#define pi 3.14159265358979323846

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
unsigned int bitrev(unsigned int n, unsigned int bits)
{
    unsigned int nrev, N;
    unsigned int count;   
    N = 1<<bits;
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


void fft_seq(complex_t* x, complex_t*y, int n)
{
  int i, m, k, j;
  int bits = (int)log2((double)n);
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
#pragma omp parallel private(i,m,k,j)
{
  //printf("%d\n", omp_get_num_threads());
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
  int size_limit = (1<<27);
  int n, i;
  struct timespec t0, t1;
  double timeint;
  srand(time(NULL));
  for(n = 8; n < size_limit; n *= 2)
  {
    double    *x0 = (double*)malloc(2*n*sizeof(double));
    complex_t *x1 = (complex_t*)malloc(n*sizeof(complex_t));
    complex_t *y1 = (complex_t*)malloc(n*sizeof(complex_t));

    for(i = 0; i < n ; ++i)
    {
      x1[i].r = x0[2*i]  = (double)rand()/RAND_MAX - 0.5;
      x1[i].i = x0[2*i+1]= (double)rand()/RAND_MAX - 0.5;
    }


    clock_gettime(CLOCK_MONOTONIC, &t0);
    fft_omp(x1, y1, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    timeint = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
    printf("problem size: %d, omp: %f\n", n, timeint);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    fft_gsl(x0, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    timeint = (t1.tv_sec + t1.tv_nsec/1e9) - (t0.tv_sec + t0.tv_nsec/1e9);
    printf("problem size: %d, gsl: %f\n", n ,timeint);


    /*
    for(i = 0; i < n; ++i)
    {
      if(fabs(1.0 - y1[i].r/x0[2*i]) > 1e-3)
        printf("%d, r: %8e, %8e\n", i, y1[i].r, x0[2*i]);
      if(fabs(1.0 - y1[i].i/x0[2*i + 1]) > 1e-3 )
        printf("%d, i: %8e, %8e\n", i, y1[i].i, x0[2*i + 1]);
    }
    */

    free(x0);
    free(x1);
    free(y1);
  }
  return 0;
}
