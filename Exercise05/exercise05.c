#include <omp.h>
#include <stdio.h>

int fib_serial(int n) {
  if (n < 2) return n;
  return fib_serial(n-1) + fib_serial(n-2);
}

int fib(int n) {
  int i, j;
  if (n < 2)
    return n;
  if (n < 20)                      // cutoff: small work is faster serially
    return fib_serial(n);
  #pragma omp task shared(i) firstprivate(n)
  i = fib(n-1);
  #pragma omp task shared(j) firstprivate(n)
  j = fib(n-2);
  #pragma omp taskwait
  return i + j;
}

int main() {
  int n = 40, result;

  double t0 = omp_get_wtime();
  int serial = fib_serial(n);
  double serial_time = omp_get_wtime() - t0;

  t0 = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp single
    result = fib(n);
  }
  double par_time = omp_get_wtime() - t0;

  printf("fib(%d) serial   = %d  time = %f s\n", n, serial, serial_time);
  printf("fib(%d) parallel = %d  time = %f s\n", n, result, par_time);
  printf("threads = %d\n", omp_get_max_threads());
  return 0;
}
