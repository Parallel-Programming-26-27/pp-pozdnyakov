
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double pi_seq(long long n)
{
    double sum = 0.0;

    for (long long i = 0; i < n; i++) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }

    return 4.0 * sum;
}


double pi_omp(long long n)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }

    return 4.0 * sum;
}

int used_threads(void)
{
    int count = 1;

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            count = omp_get_num_threads();
    }

    return count;
}

double time_seq(long long n, double *pi)
{
    double t = omp_get_wtime();
    *pi = pi_seq(n);
    t = omp_get_wtime() - t;
    return t;
}

double time_omp(long long n, int threads, double *pi)
{
    omp_set_num_threads(threads);

    double t = omp_get_wtime();
    *pi = pi_omp(n);
    t = omp_get_wtime() - t;
    return t;
}

void print_result(const char *title, long long n, double pi, double t, int threads)
{
    printf("[%s]\n", title);
    printf("  n                 = %lld\n", n);
    printf("  pi                = %.15f\n", pi);
    printf("  абсолютная ошибка = %.3e\n", fabs(pi - M_PI));
    printf("  время             = %.4f с\n", t);
    printf("  потоков           = %d\n\n", threads);
}

void experiment1(void)
{
    long long n = 1000000000LL;
    int threads[] = {1, 2, 4, 8, 16};
    double pi;

    printf("===== Эксперимент 1: n = 10^9 =====\n");

    double t_seq = time_seq(n, &pi);
    printf("Последовательная версия: %.4f с\n\n", t_seq);

    printf("Потоки  Время, с  Ускорение  Эффективность\n");
    for (int k = 0; k < 5; k++) {
        double t = time_omp(n, threads[k], &pi);
        double s = t_seq / t;
        double e = s / threads[k] * 100.0;
        printf("%6d | %8.4f | %9.2f | %11.1f %%\n", threads[k], t, s, e);
    }
    printf("\n");
}

void experiment2(void)
{
    long long sizes[] = {10000000LL, 100000000LL, 1000000000LL, 10000000000LL};
    const char *names[] = {"10^7", "10^8", "10^9", "10^10"};
    double pi;

    printf("  n     Последовательно, с OpenMP (8), с Ускорение\n");
    for (int k = 0; k < 4; k++) {
        double t_seq = time_seq(sizes[k], &pi);
        double t_omp = time_omp(sizes[k], 8, &pi);
        printf("%6s | %18.4f | %13.4f | %9.2f\n",
               names[k], t_seq, t_omp, t_seq / t_omp);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    printf("Логических ядер: %d\n\n", omp_get_num_procs());

    if (argc < 2) {
        experiment1();
        experiment2();
        return 0;
    }

    long long n = atoll(argv[1]);
    int threads = (argc >= 3) ? atoi(argv[2]) : omp_get_max_threads();

    if (n <= 0 || threads <= 0) {
        printf("Использование: %s <n> <threads>\n", argv[0]);
        return 1;
    }

    double pi_s, pi_p;

    double t_seq = time_seq(n, &pi_s);
    print_result("Последовательная версия", n, pi_s, t_seq, 1);

    double t_omp = time_omp(n, threads, &pi_p);
    print_result("OpenMP", n, pi_p, t_omp, used_threads());

    printf("Ускорение     S = T1 / Tp = %.2f\n", t_seq / t_omp);
    printf("Эффективность E = S / p   = %.1f %%\n",
           t_seq / t_omp / threads * 100.0);

    return 0;
}
