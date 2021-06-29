#include <stdlib.h>

void g(void)
{
  int* x = NULL;
  *x = 0;
}

void f(void)
{
  int a;
  g();
}             

int main(int argc, char** argv) {
  f();
  return 0;
}
