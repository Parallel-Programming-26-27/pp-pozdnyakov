#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define TAG_MESSAGE 1
#define TAG_REPLY 2
#define TAG_PING 3
#define LEN 64

void basic_exchange(int rank)
{
    char msg[LEN], reply[LEN];
    MPI_Status status;

    if (rank == 0) {
        strcpy(msg, "Hello from rank 0");
        MPI_Send(msg, LEN, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
        printf("[rank 0] sent to rank 1: \"%s\" (tag %d)\n", msg, TAG_MESSAGE);
        fflush(stdout);

        MPI_Recv(reply, LEN, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, &status);
        printf("[rank 0] received from rank %d: \"%s\" (tag %d)\n", status.MPI_SOURCE, reply, status.MPI_TAG);
        fflush(stdout);
    } else if (rank == 1) {
        MPI_Recv(msg, LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        printf("[rank 1] received from rank %d: \"%s\" (tag %d)\n", status.MPI_SOURCE, msg, status.MPI_TAG);
        fflush(stdout);

        strcpy(reply, "Message received");
        MPI_Send(reply, LEN, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
        printf("[rank 1] sent reply to rank 0: \"%s\" (tag %d)\n", reply, TAG_REPLY);
        fflush(stdout);
    }
}

void all_exchange(int rank, int size)
{
    char msg[LEN], reply[LEN], expected[LEN];
    MPI_Status status;

    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            snprintf(msg, LEN, "Task for rank %d", r);
            MPI_Send(msg, LEN, MPI_CHAR, r, TAG_MESSAGE, MPI_COMM_WORLD);
        }
        printf("[rank 0] sent %d messages\n", size - 1);
        fflush(stdout);

        int ok = 0;
        for (int r = 1; r < size; r++) {
            MPI_Recv(reply, LEN, MPI_CHAR, r, TAG_REPLY, MPI_COMM_WORLD, &status);
            snprintf(expected, LEN, "Rank %d got: Task for rank %d", r, r);
            int good = status.MPI_SOURCE == r && strcmp(reply, expected) == 0;
            ok += good;
            printf("[rank 0] reply from rank %d: \"%s\" %s\n", status.MPI_SOURCE, reply, good ? "OK" : "WRONG");
            fflush(stdout);
        }
        printf("[rank 0] sent = %d, replies = %d, all correct: %s\n", size - 1, ok, ok == size - 1 ? "yes" : "no");
        fflush(stdout);
    } else {
        MPI_Recv(msg, LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
        printf("[rank %d] received from rank %d: \"%s\"\n", rank, status.MPI_SOURCE, msg);
        fflush(stdout);

        snprintf(reply, LEN, "Rank %d got: %.40s", rank, msg);
        MPI_Send(reply, LEN, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

void ping_pong(int rank)
{
    int sizes[] = {1, 1024, 65536, 1048576};
    int reps[] = {10000, 10000, 1000, 100};
    char *buf = malloc(1048576);
    MPI_Status status;

    memset(buf, 'a', 1048576);

    for (int k = 0; k < 4; k++) {
        int n = sizes[k];
        double t = 0.0;

        for (int r = -10; r < reps[k]; r++) {
            if (r == 0)
                t = MPI_Wtime();
            if (rank == 0) {
                MPI_Send(buf, n, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD, &status);
            } else if (rank == 1) {
                MPI_Recv(buf, n, MPI_CHAR, 0, TAG_PING, MPI_COMM_WORLD, &status);
                MPI_Send(buf, n, MPI_CHAR, 0, TAG_PING, MPI_COMM_WORLD);
            }
        }
        t = MPI_Wtime() - t;

        if (rank == 0) {
            printf("size = %d bytes, reps = %d, avg round trip = %.3e s\n", n, reps[k], t / reps[k]);
            fflush(stdout);
        }
    }
    free(buf);
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("Need at least 2 processes: mpiexec -n 2 %s\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    if (argc > 1 && strcmp(argv[1], "time") == 0) {
        ping_pong(rank);
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("=== Basic exchange 0 <-> 1, processes: %d ===\n", size);
        fflush(stdout);
    }
    basic_exchange(rank);

    if (rank == 0) {
        printf("=== Exchange 0 <-> all ranks ===\n");
        fflush(stdout);
    }
    all_exchange(rank, size);

    MPI_Finalize();
    return 0;
}
