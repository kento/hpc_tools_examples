#include <stdlib.h>

void g(void)
{
  int* x = NULL;
  x[10] = 0;
}

void f(void)
{
  g();
}             

int main(void) {
  f();
  return 0;
}
