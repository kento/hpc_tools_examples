#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pi.h>
#include <mpi.h>

#include "veloc.h"

#define CKPT_PATH_FORMAT_i "./pi_i_%d.%d.ckpt"
#define CKPT_PATH_FORMAT_count "./pi_count_%d.%d.ckpt"

void print_help(){
  printf("./pi_mpi_with_veloc <restart #>  <interval> <veloc config>\n");
  printf("  <restart #>:\n");
  printf("     #=0: start from the fist interation \n");
  printf("     #=N: restart from N(>0) interation \n");
  printf("  <interval>:\n");
  printf("     If interval is set to X, it take a checkpoint every X interation\n");
  printf("  <veloc config>:\n");
  printf("     Path to veloc config file\n");
  printf(" ");
}


int main(int argc, char** argv){
  int i;
  int count=0;
  //  int start, end, n;
  //  int nprocs, myrank, tag;
  int nprocs, myrank;
  //  MPI_Status status;
  int n_sample=N_SAMPLE;
  double pi, pi_ave;
  double x, y, s;
  int restart_id;
  int interval;
 
  // MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  //  tag = 100;
  pi_ave = 0.0;
 
  if (argc != 4) {
    if (myrank == 0) print_help(); 
    MPI_Finalize();
    return 0;
  }
  fprintf(stderr, "%s\n", argv[3]);
  if (VELOC_Init(MPI_COMM_WORLD, argv[3]) != VELOC_SUCCESS) {
    printf("Error initializing VELOC! Aborting...\n");
    exit(2);
  }
  fprintf(stderr, "init done\n");
  restart_id=atoi(argv[1]);
  interval= atoi(argv[2]);  

  VELOC_Mem_protect(0, &count, 1, sizeof(int));
  int v = VELOC_Restart_test("pi", restart_id);
  if (v > 0) {
    VELOC_Restart("pi", v);
  }

  n_sample = n_sample / nprocs;

  // Checkpoiting routine
  srand(time(NULL));
  for (i = restart_id; i<n_sample; i++){ 
    if(i % interval == 0 && i != restart_id){
      VELOC_Checkpoint("pi", i);
    }
    x = (double)rand()/RAND_MAX;
    y = (double)rand()/RAND_MAX;
    s = x*x + y*y;
    if(s < 1)count++;
    if (i % (n_sample / 10) == 0 && myrank == 0) fprintf(stderr, "======= %i SAMPLING DONE ======= \n", i);
  }
  pi = (double)count / (n_sample) * 4;
  
  // MPI_Reduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&pi, &pi_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myrank == 0) printf("pi=%f\n", pi_ave /nprocs);
  VELOC_Finalize(1); // wait for checkpoints to finish   
  MPI_Finalize();
  return 0;
}
