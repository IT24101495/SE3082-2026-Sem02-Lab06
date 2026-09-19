#include <omp.h>
#include <stdio.h>
#include <math.h>

#define NPOINTS 1000
#define MAXITER 2000

struct complex { double r; double i; };

int main() {
  int numoutside = 0;
  const double eps = 1.0e-5;
  double t0 = omp_get_wtime();

  #pragma omp parallel default(none) shared(numoutside) 
  {
    struct complex z, c;
    double ztemp;
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    // block distribution of the outer loop: each thread gets an equal share of i
    int chunk = NPOINTS / nthreads;
    int start = tid * chunk;
    int end = (tid == nthreads - 1) ? NPOINTS : start + chunk;
    int local_outside = 0;

    for (int i = start; i < end; i++) {
      for (int j = 0; j < NPOINTS; j++) {
        c.r = -2.0 + 2.5 * (double)i / (double)NPOINTS + eps;
        c.i = 1.125 * (double)j / (double)NPOINTS + eps;
        z = c;
        for (int iter = 0; iter < MAXITER; iter++) {
          ztemp = z.r * z.r - z.i * z.i + c.r;
          z.i = z.r * z.i * 2 + c.i;
          z.r = ztemp;
          if (z.r * z.r + z.i * z.i > 4.0) { local_outside++; break; }
        }
      }
    }

    #pragma omp atomic
    numoutside += local_outside;
  }

  double time = omp_get_wtime() - t0;
  int numinside = NPOINTS * NPOINTS - numoutside;
  double area = 2.0 * 2.5 * 1.125 * (double)numinside / (double)(NPOINTS * NPOINTS);
  double error = area / (double)NPOINTS;

  printf("threads = %d\n", omp_get_max_threads());
  printf("Area of Mandelbrot set = %12.8f +/- %12.8f\n", area, error);
  printf("time = %f s\n", time);
  return 0;
}
