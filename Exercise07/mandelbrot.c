#include <omp.h>
#include <stdio.h>
#include <string.h>

#define NX 1200
#define NY 600
#define MAX_ITER 2000

/* The supplied lab describes starting with z = c (rather than z = 0). */
static int in_set(double cr, double ci) {
    double zr = cr, zi = ci;
    for (int it = 0; it < MAX_ITER; it++) {
        if (zr * zr + zi * zi > 4.0) return 0;
        double new_zr = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = new_zr;
    }
    return 1;
}

static long long count_row(int row) {
    long long count = 0;
    double ci = ((double)row + 0.5) * (1.5 / NY);
    for (int col = 0; col < NX; col++) {
        double cr = -2.0 + ((double)col + 0.5) * (3.0 / NX);
        count += in_set(cr, ci);
    }
    return count;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "static";
    long long inside = 0;
    double start = omp_get_wtime();

    if (strcmp(mode, "serial") == 0) {
        for (int row = 0; row < NY; row++) inside += count_row(row);
    } else if (strcmp(mode, "static") == 0) {
        /* Outer row loop divided into almost-equal blocks by thread ID. */
        #pragma omp parallel reduction(+:inside)
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();
            int begin = NY * tid / nth;
            int end = NY * (tid + 1) / nth;
            for (int row = begin; row < end; row++) inside += count_row(row);
        }
    } else if (strcmp(mode, "dynamic") == 0) {
        #pragma omp parallel for schedule(dynamic, 1) reduction(+:inside)
        for (int row = 0; row < NY; row++) inside += count_row(row);
    } else {
        fprintf(stderr, "Usage: %s [serial|static|dynamic]\n", argv[0]);
        return 1;
    }

    double elapsed_ms = (omp_get_wtime() - start) * 1000.0;
    double estimated_area = 2.0 * inside * (3.0 / NX) * (1.5 / NY);
    printf("Mode=%s; threads=%d; inside=%lld; area=%.9f; time=%.3f ms\n",
           mode, strcmp(mode, "serial") == 0 ? 1 : omp_get_max_threads(),
           inside, estimated_area, elapsed_ms);
    return 0;
}
