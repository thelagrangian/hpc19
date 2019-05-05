// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Status status;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  //srand((unsigned int) time(rank + 393919));
  srand((unsigned int) rank+time(NULL));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);

  int*lsplitters, *gsplitters, *sbeginidx, *scount, *rcount, *rbeginidx, *rarray;
  int rtotcount = 0;

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  lsplitters = (int*)malloc(sizeof(int)*(p-1));
  sbeginidx  = (int*)calloc(sizeof(int), p);
  scount     = (int*)calloc(sizeof(int), p);
  rcount     = (int*)calloc(sizeof(int), p);
  rbeginidx  = (int*)calloc(sizeof(int), p);


  for(int i = 0; i < p-1; ++i)
    lsplitters[i] = vec[(int)((double)(i+1)/p*N)];
  
  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  if(rank == 0)
    gsplitters = (int*)malloc(sizeof(int)*p*(p-1));

  MPI_Gather(lsplitters, p-1, MPI_INT, gsplitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if(rank == 0)
  {
    std::sort(gsplitters, gsplitters + p*(p-1));
    for(int i =0; i < p-1; ++i)
      lsplitters[i] = gsplitters[(i+1)*(p-1)];
  }
  // root process broadcasts splitters to all other processes
  MPI_Bcast(lsplitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);


  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  for(int i = 1; i < p; ++i)
  {
    sbeginidx[i] = std::lower_bound(vec, vec+N, lsplitters[i-1]) - vec;
    scount[i-1]  = sbeginidx[i] - sbeginidx[i-1];
  }
  scount[p-1] = N - sbeginidx[p-1];


  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data

  //MPI_Barrier(MPI_COMM_WORLD);
  MPI_Alltoall(&scount[0], 1, MPI_INT, &rcount[0], 1, MPI_INT, MPI_COMM_WORLD);
  //MPI_Barrier(MPI_COMM_WORLD);
  

  for(int i=1; i < p; ++i)
  {
    rtotcount += rcount[i-1];
    rbeginidx[i] = rtotcount;
  }
  rtotcount += rcount[p-1];

  rarray = (int*)malloc(sizeof(int)*rtotcount);

  MPI_Alltoallv(vec, scount, sbeginidx, MPI_INT, rarray, rcount, rbeginidx, MPI_INT, MPI_COMM_WORLD);


  // do a local sort of the received data
  std::sort(rarray, rarray + rtotcount);

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename,"w+");
  for(int i = 0;i< rtotcount; ++i)
    fprintf(fd, "%d\n", rarray[i]);

  fclose(fd);



  free(lsplitters);
  if(rank ==0 )
    free(gsplitters);
  free(sbeginidx);
  free(scount);
  free(rcount);
  free(rbeginidx);
  free(rarray);
  free(vec);

  MPI_Finalize();
  return 0;
}
