#include<stdio.h>
#include<stdlib.h>
#include"mpi.h"
#include<math.h>

#define MSG_SIZE (2097152/sizeof(int))

int main()
{
  int pid, commsz;
  double t0, t1;
  double tint;
  int i, j;
  int N = 100;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &commsz);

  int msg_sent[MSG_SIZE], msg_recv[MSG_SIZE];
  for(j = 0; j < MSG_SIZE; ++j)
    msg_recv[j] = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  for(i = 0; i < N; ++i)
  {
    if(pid == 0 )
    {
      msg_sent[0] = msg_recv[0] + pid;
      MPI_Send(&msg_sent, MSG_SIZE, MPI_INT, (pid + 1)%commsz, 0, MPI_COMM_WORLD);
      MPI_Recv(&msg_recv, MSG_SIZE, MPI_INT, (pid - 1 + commsz)%commsz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
      MPI_Recv(&msg_recv, MSG_SIZE, MPI_INT, (pid - 1 + commsz)%commsz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      msg_sent[0] = msg_recv[0] + pid;
      MPI_Send(&msg_sent, MSG_SIZE, MPI_INT, (pid + 1)%commsz, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  tint = t1 - t0;

  if(pid == 0)
  {
    long checksum = msg_recv[0];
    long theorsum = (long)((commsz-1)*commsz/2)*N;
    if(checksum != theorsum)
      printf("checksum error\n");
    printf("Network bandwidth: %e GB/s\n", sizeof(int)*MSG_SIZE*N/tint/1e9);
  }

  MPI_Finalize();
  return 0;
}
