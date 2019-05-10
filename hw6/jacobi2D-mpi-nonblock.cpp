/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 *
 * Updated by: Yongyan Rao
 * to solve 2D Jacobo smoothing
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  //local topology: array (lN+2)*(lN+2)

  int i,x,y;
  double tmp, gres = 0.0, lres = 0.0;

  //for (i = 1; i <= lN; i++){
  //  tmp = ((2.0*lu[i] - lu[i-1] - lu[i+1]) * invhsq - 1);
  //  lres += tmp * tmp;
  //}

  for(i = 0; i < (lN+2)*(lN+2); ++i)
  {
    x = i/(lN + 2);
    y = i%(lN + 2);
    if( x != 0 && x != (lN+1) && y != 0 && y != (lN+1) )
      tmp = ( ( 4.0*lu[i] - lu[i-1] - lu[i+1] - lu[i + lN + 2] -lu[i - lN - 2] )*invhsq - 1.0);
      lres += tmp*tmp;
  }


  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, p, N, lN, iter, max_iters;
  int pedgesize;


  MPI_Status status;
  MPI_Request rsend1, rrecv1;
  MPI_Request rsend2, rrecv2;
  MPI_Request rsend3, rrecv3;
  MPI_Request rsend4, rrecv4;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int proc_edge = (int)sqrt((double)p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  int psize_edge = (int)sqrt((double)N);

  /* compute number of unknowns handled by each process */
  lN = psize_edge / proc_edge;
  if ((psize_edge % proc_edge != 0) && mpirank == 0 ) {
    printf("problem size edge: %d, process edge: %d\n", psize_edge, proc_edge);
    printf("Exiting. problem size edge must be a multiple of process edge\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  //debug:
  //printf("problem size edge: %d, process edge: %d\n", psize_edge, proc_edge);
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN + 2));
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  int x, y;
  double *tmpsend = (double*)calloc(sizeof(double), (lN + 2));
  double *tmprecv = (double*)calloc(sizeof(double), (lN + 2));

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    //debug:
    //if(mpirank == 0)
    //  printf("iter: %d\n", iter);


    for(i = 1; i < lN + 1; ++i)
    {
      lunew[lN+2+i] =      0.25 * (hsq + lu[lN+2+i-1]      + lu[lN+2+i+1]      + lu[lN+2+i+lN+2]      + lu[lN+2+i-lN-2]);
      lunew[lN*(lN+2)+i] = 0.25 * (hsq + lu[lN*(lN+2)+i-1] + lu[lN*(lN+2)+i+1] + lu[lN*(lN+2)+i+lN+2] + lu[lN*(lN+2)+i-lN-2]);
      lunew[1+i*(lN+2)] =  0.25 * (hsq + lu[1+i*(lN+2)-1]  + lu[1+i*(lN+2)+1]  + lu[1+i*(lN+2)+lN+2]  + lu[1+i*(lN+2)-lN-2]);
      lunew[lN+i*(lN+2)] = 0.25 * (hsq + lu[lN+i*(lN+2)-1] + lu[lN+i*(lN+2)+1] + lu[lN+i*(lN+2)+lN+2] + lu[lN+i*(lN+2)-lN-2]);
    }


    int mpix = mpirank / proc_edge;
    int mpiy = mpirank % proc_edge;

    //printf("%d, mpix: %d, mpiy: %d\n", mpirank, mpix, mpiy);


    if(mpix != 0)
    {
      MPI_Isend(&lunew[lN + 2],        lN + 2, MPI_DOUBLE, mpirank - proc_edge, 123, MPI_COMM_WORLD, &rsend1);
      MPI_Irecv(&lunew[0]     ,        lN + 2, MPI_DOUBLE, mpirank - proc_edge, 124, MPI_COMM_WORLD, &rrecv1);
    }
    if(mpix != proc_edge - 1)
    {
      MPI_Isend(&lunew[lN*(lN+2)],     lN + 2, MPI_DOUBLE, mpirank + proc_edge, 124, MPI_COMM_WORLD, &rsend2);
      MPI_Irecv(&lunew[(lN+1)*(lN+2)], lN + 2, MPI_DOUBLE, mpirank + proc_edge, 123, MPI_COMM_WORLD, &rrecv2);
    }


   // if(mpirank == 0)
   //   printf("communication 1 done\n");

    if(mpiy != 0)
    {
      for(i = 0; i < lN + 2; ++i)
        tmpsend[i] = lunew[1+i*(lN+2)];

      MPI_Isend(tmpsend,               lN + 2, MPI_DOUBLE, mpirank - 1,         123, MPI_COMM_WORLD, &rsend3);
      MPI_Irecv(tmprecv,               lN + 2, MPI_DOUBLE, mpirank - 1,         124, MPI_COMM_WORLD, &rrecv3);

      for(i = 0; i < lN + 2; ++i)
        lunew[i*(lN+2)] = tmprecv[i];
    }
    if(mpiy != proc_edge - 1)
    {
      for(i = 0; i < lN + 2; ++i)
        tmpsend[i] = lunew[lN+i*(lN+2)];

      MPI_Isend(tmpsend,               lN + 2, MPI_DOUBLE, mpirank + 1,         124, MPI_COMM_WORLD, &rsend4);
      MPI_Irecv(tmprecv,               lN + 2, MPI_DOUBLE, mpirank + 1,         123, MPI_COMM_WORLD, &rrecv4);

      for(i = 0; i < lN + 2; ++i)
        lunew[lN+1+i*(lN+2)] = tmprecv[i];
    }


    /* Jacobi step for local points */
    for (i = 0; i < (lN + 2)*(lN + 2); i++)
    {
      x = i/(lN + 2);
      y = i%(lN + 2);
      if(x > 1 && x < lN && y > 1 && y < lN)
        lunew[i]  = 0.25 * (hsq + lu[i - 1] + lu[i + 1] + lu[i + lN + 2] + lu[i - lN - 2]);
    }


    if(mpix != 0)
    {
      MPI_Wait(&rsend1, &status);
      MPI_Wait(&rrecv1, &status);
    }
    if(mpix != proc_edge - 1)
    {
      MPI_Wait(&rsend2, &status);
      MPI_Wait(&rrecv2, &status);
    }
    if(mpiy != 0)
    {
      MPI_Wait(&rsend3, &status);
      MPI_Wait(&rrecv3, &status);
    }
    if(mpiy != proc_edge - 1)
    {
      MPI_Wait(&rsend4, &status);
      MPI_Wait(&rrecv4, &status);
    }



    //if(mpirank == 0)
    //  printf("communication 2 done\n");

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(tmpsend);
  free(tmprecv);
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
