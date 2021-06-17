#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
  int i;
  int count = 100;
  int n_sample = 10000000;
  double pi;
  double x, y, s;
  
  srand(time(NULL));
  for (i = 0; i < n_sample; i++) {
    x = (double)rand()/RAND_MAX;
    y = (double)rand()/RAND_MAX;
    s = x*x + y*y;
    if(s < 1) count++;

  }
  pi = (double)count / (n_sample) * 4;
  printf("%f\n", pi);

  return 0;
}
