#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static unsigned long long seed = 42;

double rnd(void)
{
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(seed >> 11) / 9007199254740992.0;
}

double soft(double z, double a)
{
    if (z > a) return z - a;
    if (z < -a) return z + a;
    return 0.0;
}

void update(double *w, double *b, const double *g, int d, double gb, double eta, double lambda)
{
    for (int j = 0; j < d; j++)
        w[j] = soft(w[j] - eta * g[j], eta * lambda);
    *b -= eta * gb;
}

void train_serial(const double *X, const double *y, int N, int d, int K,
                  double lambda, double eta, double *w, double *b)
{
    double *g = malloc(d * sizeof(double));

    for (int it = 0; it < K; it++) {
        double gb = 0.0;
        memset(g, 0, d * sizeof(double));

        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            for (int j = 0; j < d; j++)
                pred += x[j] * w[j];
            double e = pred - y[i];
            for (int j = 0; j < d; j++)
                g[j] += e * x[j];
            gb += e;
        }

        for (int j = 0; j < d; j++)
            g[j] /= N;
        update(w, b, g, d, gb / N, eta, lambda);
    }
    free(g);
}

void train_simd(const double *X, const double *y, int N, int d, int K,
                double lambda, double eta, double *w, double *b)
{
    double *g = malloc(d * sizeof(double));

    for (int it = 0; it < K; it++) {
        double gb = 0.0;
        memset(g, 0, d * sizeof(double));

        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            #pragma omp simd reduction(+:pred)
            for (int j = 0; j < d; j++)
                pred += x[j] * w[j];
            double e = pred - y[i];
            #pragma omp simd
            for (int j = 0; j < d; j++)
                g[j] += e * x[j];
            gb += e;
        }

        for (int j = 0; j < d; j++)
            g[j] /= N;
        update(w, b, g, d, gb / N, eta, lambda);
    }
    free(g);
}


void train_omp(const double *X, const double *y, int N, int d, int K,
               double lambda, double eta, double *w, double *b)
{
    int P = omp_get_max_threads();
    int stride = (d + 1 + 7) / 8 * 8;
    double *G = malloc((size_t)P * stride * sizeof(double));
    double *g = malloc((d + 1) * sizeof(double));

    for (int it = 0; it < K; it++) {
        #pragma omp parallel
        {
            int nt = omp_get_num_threads();
            double *gt = G + (size_t)omp_get_thread_num() * stride;
            for (int j = 0; j <= d; j++)
                gt[j] = 0.0;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                const double *x = X + (size_t)i * d;
                double pred = *b;
                for (int j = 0; j < d; j++)
                    pred += x[j] * w[j];
                double e = pred - y[i];
                for (int j = 0; j < d; j++)
                    gt[j] += e * x[j];
                gt[d] += e;
            }

            #pragma omp for
            for (int j = 0; j <= d; j++) {
                double s = 0.0;
                for (int t = 0; t < nt; t++)
                    s += G[(size_t)t * stride + j];
                g[j] = s / N;
            }
        }
        update(w, b, g, d, g[d], eta, lambda);
    }
    free(G);
    free(g);
}

void train_omp_simd(const double *X, const double *y, int N, int d, int K,
                    double lambda, double eta, double *w, double *b)
{
    int P = omp_get_max_threads();
    int stride = (d + 1 + 7) / 8 * 8;
    double *G = malloc((size_t)P * stride * sizeof(double));
    double *g = malloc((d + 1) * sizeof(double));

    for (int it = 0; it < K; it++) {
        #pragma omp parallel
        {
            int nt = omp_get_num_threads();
            double *gt = G + (size_t)omp_get_thread_num() * stride;
            for (int j = 0; j <= d; j++)
                gt[j] = 0.0;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                const double *x = X + (size_t)i * d;
                double pred = *b;
                #pragma omp simd reduction(+:pred)
                for (int j = 0; j < d; j++)
                    pred += x[j] * w[j];
                double e = pred - y[i];
                #pragma omp simd
                for (int j = 0; j < d; j++)
                    gt[j] += e * x[j];
                gt[d] += e;
            }

            #pragma omp for
            for (int j = 0; j <= d; j++) {
                double s = 0.0;
                for (int t = 0; t < nt; t++)
                    s += G[(size_t)t * stride + j];
                g[j] = s / N;
            }
        }
        update(w, b, g, d, g[d], eta, lambda);
    }
    free(G);
    free(g);
}

double loss(const double *X, const double *y, int N, int d, const double *w, double b, double lambda)
{
    double s = 0.0, r = 0.0;
    for (int i = 0; i < N; i++) {
        double pred = b;
        for (int j = 0; j < d; j++)
            pred += X[(size_t)i * d + j] * w[j];
        s += (pred - y[i]) * (pred - y[i]);
    }
    for (int j = 0; j < d; j++)
        r += fabs(w[j]);
    return s / (2.0 * N) + lambda * r;
}

typedef void (*train_fn)(const double *, const double *, int, int, int, double, double, double *, double *);

int cmp(const void *p, const void *q)
{
    double a = *(const double *)p, c = *(const double *)q;
    return (a > c) - (a < c);
}

int main(int argc, char *argv[])
{
    int N = 100000, d = 32, K = 100, reps = 3;
    double lambda = 1e-3, eta = 0.5;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) d = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) lambda = atof(argv[4]);
    if (argc > 5) eta = atof(argv[5]);
    if (argc > 6) omp_set_num_threads(atoi(argv[6]));
    if (argc > 7) reps = atoi(argv[7]);

    double *X = malloc((size_t)N * d * sizeof(double));
    double *y = malloc(N * sizeof(double));
    double *wt = malloc(d * sizeof(double));
    double *w = malloc(d * sizeof(double));
    double *ws = malloc(d * sizeof(double));
    double *times = malloc(reps * sizeof(double));

    for (int j = 0; j < d; j++)
        wt[j] = (j % 4 == 0) ? 0.0 : 2.0 * rnd() - 1.0;
    for (size_t k = 0; k < (size_t)N * d; k++)
        X[k] = 2.0 * rnd() - 1.0;
    for (int i = 0; i < N; i++) {
        double s = 0.5;
        for (int j = 0; j < d; j++)
            s += X[(size_t)i * d + j] * wt[j];
        y[i] = s + 0.1 * (rnd() - 0.5);
    }

    const char *names[] = {"serial", "omp", "simd", "omp_simd"};
    train_fn funcs[] = {train_serial, train_omp, train_simd, train_omp_simd};
    double Fs = 0.0;

    printf("N=%d d=%d K=%d lambda=%g eta=%g threads=%d reps=%d\n",
           N, d, K, lambda, eta, omp_get_max_threads(), reps);

    for (int v = 0; v < 4; v++) {
        double b = 0.0;
        for (int r = -1; r < reps; r++) {
            memset(w, 0, d * sizeof(double));
            b = 0.0;
            double t = omp_get_wtime();
            funcs[v](X, y, N, d, K, lambda, eta, w, &b);
            t = omp_get_wtime() - t;
            if (r >= 0)
                times[r] = t;
        }
        qsort(times, reps, sizeof(double), cmp);

        double F = loss(X, y, N, d, w, b, lambda);
        int zeros = 0;
        double dw = 0.0;
        if (v == 0) {
            memcpy(ws, w, d * sizeof(double));
            Fs = F;
        }
        for (int j = 0; j < d; j++) {
            if (fabs(w[j]) < 1e-6) zeros++;
            if (fabs(w[j] - ws[j]) > dw) dw = fabs(w[j] - ws[j]);
        }
        int ok = dw < 1e-9 && fabs(F - Fs) < 1e-9;

        printf("%-9s time=%.6f s  F=%.10f  dw=%.1e  zeros=%d/%d  check=%s\n",
               names[v], times[reps / 2], F, dw, zeros, d, ok ? "OK" : "FAIL");
    }

    free(X);
    free(y);
    free(wt);
    free(w);
    free(ws);
    free(times);
    return 0;
}
