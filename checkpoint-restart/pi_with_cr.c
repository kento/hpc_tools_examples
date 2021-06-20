#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pi.h>

#define CKPT_PATH_FORMAT_i "./pi_i.%d.ckpt"
#define CKPT_PATH_FORMAT_count "./pi_count.%d.ckpt"

int print_help() {
  printf("./pi_with_cr <restart #> <interval>\n");
  printf("   <restart #>:\n");
  printf("      #=0: start from the fist iteration \n");
  printf("      #=N: restart from N(>0) iteratione \n");
  printf("   <interval>:\n");
  printf("      If interval is set to X, it take a checkpoint every X iteration\n");
  printf("");
  exit(0);
}

int main(int argc, char** argv) {
  int i;
  int count = 0;
  int n_sample = N_SAMPLE;
  double pi;
  double x, y, s;
  int restart_id;
  int interval; /**/
  int fd_i, fd_count;
  char ckpt_name_i[32];
  char ckpt_name_count[32];
  int mode;
  int ret;

  if (argc != 3) print_help();
  restart_id = atoi(argv[1]);
  interval = atoi(argv[2]);
  mode  = S_IRUSR | S_IWUSR;

  /* Begin: Restart Routine */
  if (restart_id > 0) {
    sprintf(ckpt_name_count, CKPT_PATH_FORMAT_count, restart_id);
    fd_count = open(ckpt_name_count, O_RDONLY, mode);
    if (fd_count < 0) fprintf(stderr, "Error at open\n");
    ret = read(fd_count, &count, sizeof(int));
    if (ret < 0) fprintf(stderr, "Error at read\n");
    ret = close(fd_count);
    if (ret < 0) fprintf(stderr, "Error at close\n");
    printf("Restarted from iteration %d completed (count: %d)\n", restart_id, count);
  }
  /* End: Restart Routine */
  
  srand(time(NULL));
  for (i = restart_id; i < n_sample; i++) {
    /* Begin: Checkpoint Routine */
    if (i % interval == 0 && i != restart_id) {
      sprintf(ckpt_name_count, CKPT_PATH_FORMAT_count, i);
      fd_count = open(ckpt_name_count, O_WRONLY | O_CREAT | O_TRUNC, mode);
      ret = write(fd_count, &count, sizeof(int));
      if (ret < 0) fprintf(stderr, "Error at write\n");
      ret = close(fd_count);
      if (ret < 0) fprintf(stderr, "Error at close\n");
      printf("Checkpoint at iteration %d completed (count: %d)\n", i, count);
    }
    /* End: Checkpoint Routine */
    x = (double)rand()/RAND_MAX;
    y = (double)rand()/RAND_MAX;
    s = x*x + y*y;
    if(s < 1) count++;
    if (i % (n_sample / 10) == 0 ) printf("%i sampling done\n", i);

  }
  pi = (double)count / (n_sample) * 4;
  printf("%f\n", pi);

  return 0;
}
