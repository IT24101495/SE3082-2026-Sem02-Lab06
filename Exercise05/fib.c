#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define TASK_CUTOFF 25

static int fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n - 1) + fib_serial(n - 2);
}

static int fib_parallel(int n) {
    if (n <= TASK_CUTOFF) return fib_serial(n);

    int a = 0, b = 0;

    #pragma omp task shared(a) firstprivate(n)
    a = fib_parallel(n - 1);

    #pragma omp task shared(b) firstprivate(n)
    b = fib_parallel(n - 2);

    #pragma omp taskwait
    return a + b;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 38;
    if (n < 0 || n > 46) {
        fprintf(stderr, "Please enter n between 0 and 46.\n");
        return 1;
    }

    double start = omp_get_wtime();
    int serial_result = fib_serial(n);
    double serial_ms = (omp_get_wtime() - start) * 1000.0;

    int parallel_result = 0;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        parallel_result = fib_parallel(n);
    }
    double parallel_ms = (omp_get_wtime() - start) * 1000.0;

    printf("Fibonacci n = %d\n", n);
    printf("Serial result   = %d, time = %.3f ms\n", serial_result, serial_ms);
    printf("Parallel result = %d, time = %.3f ms (%d requested threads)\n",
           parallel_result, parallel_ms, omp_get_max_threads());
    printf("Correct: %s\n", serial_result == parallel_result ? "YES" : "NO");
    return serial_result == parallel_result ? 0 : 1;
}
