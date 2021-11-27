#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pi.h>
#include <mpi.h>

#define CKPT_PATH_FORMAT_i "./pi_i_%d.%d.ckpt"
#define CKPT_PATH_FORMAT_count "./pi_count_%d.%d.ckpt"

void print_help(){
  printf("./pi_mpi_with_cr <restart #>  <interval>\n");
  printf("  <restart #>:\n");
  printf("     #=0: start from the fist interation \n");
  printf("     #=N: restart from N(>0) interation \n");
  printf("   <interval>:\n");
  printf("     If interval is set to X, it take a checkpoint every X interation\n");
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
  // int fd_i, fd_count;
  int fd_count;
 
  char ckpt_name_count[32];
  int mode;
  int ret;
 
 
  // MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  //  tag = 100;
  pi_ave = 0.0;
 
  if (argc != 3) {
    if (myrank == 0) print_help(); // hikisuu 3 aruka kakuninn
    MPI_Finalize();
    return 0;
  }
  restart_id=atoi(argv[1]);
  interval= atoi(argv[2]);   //2th hikisuu
  mode = S_IRUSR | S_IWUSR;
 
 
  // Restart routine
  if (restart_id>0){
    sprintf(ckpt_name_count, CKPT_PATH_FORMAT_count,myrank, restart_id);  //pi_count_0.0.ckpt    pi_count_1.0.ckpt
    fd_count=open(ckpt_name_count, O_RDONLY, mode);
    if (fd_count<0) fprintf(stderr, "Error at open\n");
    ret=read(fd_count, &count, sizeof(int));
    if (ret<0) fprintf(stderr, "ERROR at close\n");
    ret=close(fd_count);
    if(ret<0)fprintf(stderr,"ERROR at close\n");
    printf(" Restarted from interation %d completed (count: %d)\n", restart_id,count);
  }


  // Checkpoiting routine
  srand(time(NULL));
  for (i = restart_id; i<n_sample; i++){    //hajimedake joagi
    if(i % interval == 0 && i != restart_id){
      sprintf(ckpt_name_count, CKPT_PATH_FORMAT_count, myrank, i);
      fd_count =open(ckpt_name_count, O_WRONLY | O_CREAT | O_TRUNC, mode);  //write, make file,size 0
      ret = write(fd_count,&count, sizeof(int)); // integer size kannsuu mttekuru
      if(ret<0) fprintf(stderr, "Error at write\n");
      ret=close(fd_count);
      if (ret < 0) fprintf(stderr, "Error at close\n");
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
  MPI_Finalize();
  return 0;
}
