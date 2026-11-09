#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    long long N = 0;
    int reps = 10;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc > 1)
            N = atoll(argv[1]);
        if (argc > 2)
            reps = atoi(argv[2]);
    }

    MPI_Bcast(&N, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N <= 0 || N > 1000000000LL || reps <= 0) {
        if (rank == 0)
            printf("Usage: mpiexec -n P %s N [reps], N > 0\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int r = 0; r < size; r++) {
        counts[r] = (int)(N / size + (r < N % size ? 1 : 0));
        displs[r] = r == 0 ? 0 : displs[r - 1] + counts[r - 1];
    }
    int even = N % size == 0;
    int local_n = counts[rank];

    int *a = NULL;
    if (rank == 0) {
        a = malloc(N * sizeof(int));
        for (long long i = 0; i < N; i++)
            a[i] = (int)(i + 1);
    }
    int *local = malloc((local_n > 0 ? local_n : 1) * sizeof(int));
    long long *all_sums = malloc(size * sizeof(long long));

    long long sum_gather = 0, sum_reduce = 0, sum_allreduce = 0, local_sum = 0;
    double t_total = 0.0;

    for (int it = -1; it < reps; it++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime();

        if (even)
            MPI_Scatter(a, local_n, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);
        else
            MPI_Scatterv(a, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        local_sum = 0;
        for (int i = 0; i < local_n; i++)
            local_sum += local[i];

        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, all_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            sum_gather = 0;
            for (int r = 0; r < size; r++)
                sum_gather += all_sums[r];
        }

        MPI_Reduce(&local_sum, &sum_reduce, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sum, &sum_allreduce, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

        t = MPI_Wtime() - t;
        if (it >= 0)
            t_total += t;
    }

    double t_max;
    MPI_Reduce(&t_total, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long expected = N * (N + 1) / 2;

    if (rank == 0) {
        printf("P = %d, N = %lld, distribution: %s\n", size, N, even ? "MPI_Scatter" : "MPI_Scatterv");
        printf("expected  = %lld\n", expected);
        printf("Gather    = %lld\n", sum_gather);
        printf("Reduce    = %lld\n", sum_reduce);
        printf("Allreduce = %lld\n", sum_allreduce);
        printf("correct: %s\n", sum_gather == expected && sum_reduce == expected && sum_allreduce == expected ? "yes" : "no");
        printf("avg time = %.6f s (reps = %d)\n", t_max / reps, reps);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double share = 100.0 * local_sum / sum_allreduce;
    printf("[rank %d] elements = %d, local sum = %lld, total (Allreduce) = %lld, share = %.2f %%\n",
           rank, local_n, local_sum, sum_allreduce, share);
    fflush(stdout);

    int agg_reps = 100000;
    long long s1 = 0, s2 = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double tg = MPI_Wtime();
    for (int it = 0; it < agg_reps; it++) {
        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, all_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            s1 = 0;
            for (int r = 0; r < size; r++)
                s1 += all_sums[r];
        }
    }
    tg = MPI_Wtime() - tg;

    MPI_Barrier(MPI_COMM_WORLD);
    double tr = MPI_Wtime();
    for (int it = 0; it < agg_reps; it++)
        MPI_Reduce(&local_sum, &s2, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    tr = MPI_Wtime() - tr;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Gather + sum: %.3e s per op (reps = %d), correct: %s\n", tg / agg_reps, agg_reps, s1 == expected ? "yes" : "no");
        printf("Reduce:       %.3e s per op (reps = %d), correct: %s\n", tr / agg_reps, agg_reps, s2 == expected ? "yes" : "no");
        fflush(stdout);
    }

    free(a);
    free(local);
    free(all_sums);
    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}
