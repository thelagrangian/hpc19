#include<stdio.h>
#include<stdlib.h>
#include"mpi.h"
#include<math.h>

#define MSG_SIZE 1

int main()
{
  int pid, commsz;
  double t0, t1;
  double tint;
  int i;
  int N = 1000;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &commsz);

  int msg_sent[MSG_SIZE], msg_recv[MSG_SIZE];
  msg_recv[0] = 0;

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
    long theorsum = (long)(commsz-1)*commsz/2*N;
    printf("Expected sum: %d\n", theorsum);
    printf("Ring sum: %d\nTime latency: %f ms\n", msg_recv[0], tint/N/commsz*1000);
  }

  MPI_Finalize();
  return 0;
}
