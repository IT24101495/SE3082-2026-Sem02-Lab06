#include <omp.h>
#include <stdio.h>
#include <math.h>

#define N 1000000
#define STRIP 16            // 16 floats = 64 bytes = one cache line, multiple of SIMD width
#define REPEAT 20

static float A[N], B[N], C[N], Cs[N];

int main() {
  for (int i = 0; i < N; i++) {
    A[i] = (float)(i % 100) + 1.0f;
    B[i] = (float)(i % 50) + 2.0f;
  }

  // serial baseline
  double t0 = omp_get_wtime();
  for (int rep = 0; rep < REPEAT; rep++)
    for (int i = 0; i < N; i++)
      Cs[i] = A[i] * B[i];
  double serial = (omp_get_wtime() - t0) / REPEAT;

  // parallel + strip mining + simd
  t0 = omp_get_wtime();
  for (int rep = 0; rep < REPEAT; rep++) {
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < N; s += STRIP) {
      int end = (s + STRIP < N) ? s + STRIP : N;
      #pragma omp simd
      for (int i = s; i < end; i++)
        C[i] = A[i] * B[i];
    }
  }
  double par = (omp_get_wtime() - t0) / REPEAT;

  int ok = 1;
  for (int i = 0; i < N; i++)
    if (C[i] != Cs[i]) { ok = 0; break; }

  printf("threads = %d, strip = %d\n", omp_get_max_threads(), STRIP);
  printf("serial   = %f ms\n", serial * 1000);
  printf("parallel = %f ms\n", par * 1000);
  printf("speedup  = %.2f, results %s\n", serial / par, ok ? "MATCH" : "DIFFER");
  return 0;
}
