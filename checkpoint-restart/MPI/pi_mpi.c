#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pi.h>
#include <mpi.h>


int main(int argc, char** argv){
  int i;
  int count=0;

  int nprocs, myrank;
  int n_sample=N_SAMPLE;
  double pi, pi_ave;
  double x, y, s;
 
  // MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  pi_ave = 0.0;

  n_sample = n_sample / nprocs;

  srand(time(NULL));
  for (i = 0; i<n_sample; i++){    //hajimedake joagi
    x = (double)rand()/RAND_MAX;
    y = (double)rand()/RAND_MAX;
    s = x*x + y*y;
    if(s < 1)count++;
    if (i % (n_sample / 10) == 0 && myrank == 0) fprintf(stderr, "======= %i SAMPLING DONE ======= \n", i);
  }
  pi = (double)count / (n_sample) * 4;

  MPI_Reduce(&pi, &pi_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myrank == 0) printf("pi=%f\n", pi_ave /nprocs);
  MPI_Finalize();
  return 0;
}
