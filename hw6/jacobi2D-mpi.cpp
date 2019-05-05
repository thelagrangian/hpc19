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


  MPI_Status status, status1;

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

    /* Jacobi step for local points */
    for (i = 0; i < (lN + 2)*(lN + 2); i++){
      x = i/(lN + 2);
      y = i%(lN + 2);
      if(x != 0 && x != (lN + 1) && y != 0 && y != (lN + 1))
        lunew[i]  = 0.25 * (hsq + lu[i - 1] + lu[i + 1] + lu[i + lN + 2] + lu[i - lN - 2]);
    }

    /* communicate ghost values */
    //if (mpirank < p - 1) {
    //  /* If not the last process, send/recv bdry values to the right */
    //  MPI_Send(&(lunew[lN]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
    //  MPI_Recv(&(lunew[lN+1]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
    //}
    //if (mpirank > 0) {
    //  /* If not the first process, send/recv bdry values to the left */
    //  MPI_Send(&(lunew[1]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
    //  MPI_Recv(&(lunew[0]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    //}

    int mpix = mpirank / proc_edge;
    int mpiy = mpirank % proc_edge;
    if(mpix != 0)
    {
      MPI_Send(&lunew[lN + 2],        lN + 2, MPI_DOUBLE, mpirank - proc_edge, 123, MPI_COMM_WORLD);
      MPI_Recv(&lunew[0]     ,        lN + 2, MPI_DOUBLE, mpirank - proc_edge, 124, MPI_COMM_WORLD, &status);
    }
    if(mpix != proc_edge - 1)
    {
      MPI_Send(&lunew[lN*(lN+2)],     lN + 2, MPI_DOUBLE, mpirank + proc_edge, 124, MPI_COMM_WORLD);
      MPI_Recv(&lunew[(lN+1)*(lN+2)], lN + 2, MPI_DOUBLE, mpirank + proc_edge, 123, MPI_COMM_WORLD, &status);
    }

    if(mpiy != 0)
    {
      for(i = 0; i < lN + 2; ++i)
        tmpsend[i] = lunew[1+i*(lN+2)];

      MPI_Send(tmpsend,               lN + 2, MPI_DOUBLE, mpirank - 1,         123, MPI_COMM_WORLD);
      MPI_Recv(tmprecv,               lN + 2, MPI_DOUBLE, mpirank - 1,         124, MPI_COMM_WORLD, &status);

      for(i = 0; i < lN + 2; ++i)
        lunew[i*(lN+2)] = tmprecv[i];
    }
    if(mpiy != proc_edge - 1)
    {
      for(i = 0; i < lN + 2; ++i)
        tmpsend[i] = lunew[lN+i*(lN+2)];

      MPI_Send(tmpsend,               lN + 2, MPI_DOUBLE, mpirank + 1,         124, MPI_COMM_WORLD);
      MPI_Recv(tmprecv,               lN + 2, MPI_DOUBLE, mpirank + 1,         123, MPI_COMM_WORLD, &status);

      for(i = 0; i < lN + 2; ++i)
        lunew[lN+1+i*(lN+2)] = tmprecv[i];
    }


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
