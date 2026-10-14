#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x)
{
    return sqrt(2.0 + sin(x));
}

double integral_seq(double a, double b, long long n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (long long i = 0; i < n; i++)
        sum += f(a + (i + 0.5) * h);

    return sum * h;
}

double integral_omp(double a, double b, long long n, int threads)
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) num_threads(threads)
    for (long long i = 0; i < n; i++)
        sum += f(a + (i + 0.5) * h);

    return sum * h;
}

long long find_n(double a, double b, double eps, int threads, double *res)
{
    long long n = 10;
    double prev, cur;

    if (threads == 1)
        prev = integral_seq(a, b, n);
    else
        prev = integral_omp(a, b, n, threads);

    while (1) {
        n *= 2;
        if (threads == 1)
            cur = integral_seq(a, b, n);
        else
            cur = integral_omp(a, b, n, threads);
        if (fabs(cur - prev) < eps)
            break;
        prev = cur;
    }

    *res = cur;
    return n;
}

int main(int argc, char *argv[])
{
    double a = 0.0;
    double b = M_PI / 2;
    long long n = 100000000;
    int threads = 8;
    double eps = 0.0;
    double t, I;

    if (argc > 1) a = atof(argv[1]);
    if (argc > 2) b = atof(argv[2]);
    if (argc > 3) n = atoll(argv[3]);
    if (argc > 4) threads = atoi(argv[4]);
    if (argc > 5) eps = atof(argv[5]);

    if (eps > 0) {
        t = omp_get_wtime();
        n = find_n(a, b, eps, 1, &I);
        t = omp_get_wtime() - t;
        printf("seq: n = %lld  I = %.12f  time = %.6f s  threads = 1\n", n, I, t);

        t = omp_get_wtime();
        n = find_n(a, b, eps, threads, &I);
        t = omp_get_wtime() - t;
        printf("omp: n = %lld  I = %.12f  time = %.6f s  threads = %d\n", n, I, t, threads);
        return 0;
    }

    t = omp_get_wtime();
    I = integral_seq(a, b, n);
    t = omp_get_wtime() - t;
    printf("seq: n = %lld  I = %.12f  time = %.6f s  threads = 1\n", n, I, t);

    t = omp_get_wtime();
    I = integral_omp(a, b, n, threads);
    t = omp_get_wtime() - t;
    printf("omp: n = %lld  I = %.12f  time = %.6f s  threads = %d\n", n, I, t, threads);

    return 0;
}
