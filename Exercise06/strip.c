#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define STRIP_SIZE 64
#define REPEATS 100

int main(void) {
    float *A = malloc((size_t)N * sizeof(float));
    float *B = malloc((size_t)N * sizeof(float));
    float *Cserial = malloc((size_t)N * sizeof(float));
    float *Cparallel = malloc((size_t)N * sizeof(float));
    if (!A || !B || !Cserial || !Cparallel) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(A); free(B); free(Cserial); free(Cparallel);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i % 100) * 0.5f;
        B[i] = (float)(i % 50) * 0.25f;
    }

    double start = omp_get_wtime();
    for (int rep = 0; rep < REPEATS; rep++) {
        for (int i = 0; i < N; i++) {
            Cserial[i] = A[i] * B[i];
        }
    }
    double serial_ms = (omp_get_wtime() - start) * 1000.0;

    start = omp_get_wtime();
    for (int rep = 0; rep < REPEATS; rep++) {
        #pragma omp parallel for schedule(static)
        for (int base = 0; base < N; base += STRIP_SIZE) {
            int end = base + STRIP_SIZE;
            if (end > N) end = N;

            #pragma omp simd
            for (int i = base; i < end; i++) {
                Cparallel[i] = A[i] * B[i];
            }
        }
    }
    double parallel_ms = (omp_get_wtime() - start) * 1000.0;

    int errors = 0;
    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        if (fabsf(Cserial[i] - Cparallel[i]) > 1e-6f) errors++;
        checksum += Cparallel[i];
    }

    printf("N=%d, strip=%d floats, repeats=%d, threads=%d\n",
           N, STRIP_SIZE, REPEATS, omp_get_max_threads());
    printf("Serial time   = %.3f ms\n", serial_ms);
    printf("Parallel time = %.3f ms\n", parallel_ms);
    printf("Speedup       = %.3f\n", serial_ms / parallel_ms);
    printf("Mismatches    = %d\n", errors);
    printf("Checksum      = %.6f\n", checksum);

    free(A); free(B); free(Cserial); free(Cparallel);
    return errors == 0 ? 0 : 1;
}
