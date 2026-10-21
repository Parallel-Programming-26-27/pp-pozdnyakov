#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void ijk_seq(const double *A, const double *B, double *C, int N, int bs)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void ijk_omp(const double *A, const double *B, double *C, int N, int bs)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void ikj_seq(const double *A, const double *B, double *C, int N, int bs)
{
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

void ikj_omp(const double *A, const double *B, double *C, int N, int bs)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

void blocked_omp(const double *A, const double *B, double *C, int N, int bs)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += bs)
        for (int jj = 0; jj < N; jj += bs)
            for (int kk = 0; kk < N; kk += bs) {
                int imax = ii + bs < N ? ii + bs : N;
                int jmax = jj + bs < N ? jj + bs : N;
                int kmax = kk + bs < N ? kk + bs : N;
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        double a = A[i * N + k];
                        for (int j = jj; j < jmax; j++)
                            C[i * N + j] += a * B[k * N + j];
                    }
            }
}

typedef void (*mul_fn)(const double *, const double *, double *, int, int);

int cmp(const void *x, const void *y)
{
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

int main(int argc, char *argv[])
{
    int N = 512;
    int bs = 32;
    int reps = 5;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) bs = atoi(argv[2]);
    if (argc > 3) omp_set_num_threads(atoi(argv[3]));
    if (argc > 4) reps = atoi(argv[4]);

    size_t size = (size_t)N * N;
    double *A = malloc(size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *C = malloc(size * sizeof(double));
    double *R = malloc(size * sizeof(double));
    double *times = malloc(reps * sizeof(double));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)((i * 7 + j * 3) % 19) / 10.0 - 0.9;
            B[i * N + j] = (double)((i * 5 + j * 11) % 23) / 10.0 - 1.1;
        }

    memset(R, 0, size * sizeof(double));
    ijk_seq(A, B, R, N, bs);

    const char *names[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};
    mul_fn funcs[] = {ijk_seq, ijk_omp, ikj_seq, ikj_omp, blocked_omp};

    printf("N=%d bs=%d threads=%d reps=%d\n", N, bs, omp_get_max_threads(), reps);

    for (int v = 0; v < 5; v++) {
        double maxerr = 0.0;

        for (int r = -1; r < reps; r++) {
            memset(C, 0, size * sizeof(double));

            double t = omp_get_wtime();
            funcs[v](A, B, C, N, bs);
            t = omp_get_wtime() - t;

            if (r < 0)
                continue;
            times[r] = t;
            for (size_t q = 0; q < size; q++) {
                double e = fabs(C[q] - R[q]) / (fabs(R[q]) + 1.0);
                if (e > maxerr)
                    maxerr = e;
            }
        }

        qsort(times, reps, sizeof(double), cmp);
        printf("%-12s time=%.6f s  maxerr=%.1e  check=%s\n",
               names[v], times[reps / 2], maxerr, maxerr < 1e-9 ? "OK" : "FAIL");
    }

    free(A);
    free(B);
    free(C);
    free(R);
    free(times);
    return 0;
}
