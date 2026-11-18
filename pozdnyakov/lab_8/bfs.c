#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int from;
    int to;
} Edge;

typedef struct {
    int vertices;
    int edge_count;
    int *offsets;
    int *neighbors;
} Graph;

typedef struct {
    int *items;
    size_t size;
    size_t capacity;
} List;

static void fail(const char *message)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Процесс %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t size)
{
    if (count == 0) {
        count = 1;
    }
    if (count > SIZE_MAX / size) {
        fail("Запрошен слишком большой объём памяти");
    }
    void *memory = calloc(count, size);
    if (memory == NULL) {
        fail("Недостаточно памяти");
    }
    return memory;
}

static int number(const char *text, int min, int max)
{
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno != 0 || value < min || value > max) {
        fail("Числовой аргумент некорректен или вне допустимого диапазона");
    }
    return (int)value;
}

static Graph build_csr(int n, int m, const Edge *edges, int directed)
{
    Graph graph = {n, m, NULL, NULL};
    int *offsets = allocate((size_t)n + 1, sizeof(int));
    int *cursor = allocate(n, sizeof(int));

    for (int i = 0; i < m; ++i) {
        ++offsets[edges[i].from + 1];
        if (!directed) {
            ++offsets[edges[i].to + 1];
        }
    }
    for (int v = 0; v < n; ++v) {
        offsets[v + 1] += offsets[v];
        cursor[v] = offsets[v];
    }

    graph.offsets = offsets;
    graph.neighbors = allocate(offsets[n], sizeof(int));
    for (int i = 0; i < m; ++i) {
        int from = edges[i].from;
        int to = edges[i].to;
        graph.neighbors[cursor[from]++] = to;
        if (!directed) {
            graph.neighbors[cursor[to]++] = from;
        }
    }
    free(cursor);
    return graph;
}

static Graph read_graph(const char *path, int directed, int rank)
{
    int n = 0;
    int m = 0;
    int *ends = NULL;
    if (rank == 0) {
        FILE *input = fopen(path, "r");
        if (input == NULL) {
            fail("Не удалось открыть файл графа");
        }
        if (fscanf(input, "%d%d", &n, &m) != 2 || n < 1 || n == INT_MAX ||
            m < 0 || m > INT_MAX / 2) {
            fail("Некорректный заголовок файла графа");
        }
        ends = allocate((size_t)2 * m, sizeof(int));
        for (int i = 0; i < m; ++i) {
            int *from = &ends[2 * i];
            int *to = &ends[2 * i + 1];
            if (fscanf(input, "%d%d", from, to) != 2 ||
                *from < 0 || *from >= n || *to < 0 || *to >= n) {
                fail("Некорректное ребро в файле графа");
            }
        }
        fclose(input);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        ends = allocate((size_t)2 * m, sizeof(int));
    }
    MPI_Bcast(ends, 2 * m, MPI_INT, 0, MPI_COMM_WORLD);

    Edge *edges = allocate(m, sizeof(Edge));
    for (int i = 0; i < m; ++i) {
        edges[i] = (Edge){ends[2 * i], ends[2 * i + 1]};
    }
    free(ends);
    Graph graph = build_csr(n, m, edges, directed);
    free(edges);
    return graph;
}

static uint32_t random_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static Graph generate_graph(int n, int extra, const char *kind)
{
    int skewed = strcmp(kind, "skew") == 0;
    if (!skewed && strcmp(kind, "ring") != 0) {
        fail("Укажите тип графа ring или skew");
    }
    if ((long long)n * (1LL + extra) > INT_MAX / 2) {
        fail("Размер генерируемого графа превышает допустимый");
    }
    int m = n * (1 + extra);
    Edge *edges = allocate(m, sizeof(Edge));
    uint32_t seed = 42;
    int hubs = n / 100 + 1;

    /* Кольцо делает граф связным. Дополнительные рёбра сокращают пути. */
    for (int v = 0; v < n; ++v) {
        edges[v] = (Edge){v, (v + 1) % n};
    }
    for (int i = n; i < m; ++i) {
        int from = (i - n) % n;
        int limit = skewed ? hubs : n;
        int to = (int)(random_next(&seed) % (uint32_t)limit);
        edges[i] = (Edge){from, to};
    }
    Graph graph = build_csr(n, m, edges, 0);
    free(edges);
    return graph;
}

static void reset_distances(int *dist, int n, int start)
{
    for (int v = 0; v < n; ++v) {
        dist[v] = -1;
    }
    dist[start] = 0;
}

static void bfs_sequential(const Graph *graph, int start, int *dist, int *queue)
{
    reset_distances(dist, graph->vertices, start);
    int head = 0;
    int tail = 1;
    queue[0] = start;
    while (head < tail) {
        int v = queue[head++];
        for (int j = graph->offsets[v]; j < graph->offsets[v + 1]; ++j) {
            int u = graph->neighbors[j];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
}

static int append(List *list, int value)
{
    if (list->size == list->capacity) {
        if (list->capacity > SIZE_MAX / sizeof(int) / 2) {
            return 0;
        }
        size_t capacity = list->capacity == 0 ? 64 : 2 * list->capacity;
        int *items = realloc(list->items, capacity * sizeof(int));
        if (items == NULL) {
            return 0;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->size++] = value;
    return 1;
}

static int bfs_hybrid(const Graph *graph, int start, int *dist,
                      int rank, int nprocs, int *frontier, int *next,
                      int *local, int *global, List *lists, int threads,
                      int *sizes, int *processed, int *actual_threads)
{
    int n = graph->vertices;
    int size = 1;
    int level = 0;
    *processed = 0;
    reset_distances(dist, n, start);
    frontier[0] = start;

    while (size > 0) {
        if (sizes != NULL) {
            sizes[level] = size;
        }
        int own = 0;
        if (rank < size) {
            own = 1 + (size - 1 - rank) / nprocs;
        }
        *processed += own;
        int allocation_failed = 0;

        /* Поток пишет только в свой список и читает общий массив расстояний. */
        #pragma omp parallel reduction(|:allocation_failed)
        {
            int tid = omp_get_thread_num();
            List *list = &lists[tid];
            #pragma omp master
            {
                *actual_threads = omp_get_num_threads();
            }
            #pragma omp for schedule(runtime)
            for (int k = 0; k < own; ++k) {
                int v = frontier[rank + k * nprocs];
                for (int j = graph->offsets[v]; j < graph->offsets[v + 1]; ++j) {
                    int u = graph->neighbors[j];
                    if (dist[u] == -1 && !allocation_failed && !append(list, u)) {
                        allocation_failed = 1;
                    }
                }
            }
        }
        if (allocation_failed) {
            fail("Не удалось увеличить список кандидатов");
        }

        memset(local, 0, (size_t)n * sizeof(int));
        for (int t = 0; t < threads; ++t) {
            for (size_t i = 0; i < lists[t].size; ++i) {
                local[lists[t].items[i]] = 1;
            }
            lists[t].size = 0;
        }
        MPI_Allreduce(local, global, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int next_size = 0;
        for (int u = 0; u < n; ++u) {
            if (global[u] && dist[u] == -1) {
                dist[u] = level + 1;
                next[next_size++] = u;
            }
        }
        int *temp = frontier;
        frontier = next;
        next = temp;
        size = next_size;
        ++level;
    }
    return level;
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fail("Требуется поддержка MPI_THREAD_FUNNELED");
    }
    int rank;
    int nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Запуск: bfs файл_графа начальная_вершина [--directed] [--stats]\n"
                            "        bfs --generate вершины доп_рёбер_на_вершину ring|skew начальная_вершина [--stats]\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int generated = strcmp(argv[1], "--generate") == 0;
    int start_index = generated ? 5 : 2;
    if (argc <= start_index) {
        fail("Не указана начальная вершина или параметры генерации графа");
    }

    int stats = 0;
    int directed = 0;
    for (int i = start_index + 1; i < argc; ++i) {
        if (strcmp(argv[i], "--stats") == 0) {
            stats = 1;
        } else if (!generated && strcmp(argv[i], "--directed") == 0) {
            directed = 1;
        } else {
            fail("Неизвестный параметр запуска");
        }
    }

    Graph graph;
    if (generated) {
        int n = number(argv[2], 2, INT_MAX - 1);
        int extra = number(argv[3], 0, INT_MAX / 2);
        graph = generate_graph(n, extra, argv[4]);
    } else {
        graph = read_graph(argv[1], directed, rank);
    }

    int n = graph.vertices;
    int start = number(argv[start_index], 0, n - 1);
    omp_set_dynamic(0);
    int threads = omp_get_max_threads();
    if (getenv("OMP_SCHEDULE") == NULL) {
        omp_set_schedule(omp_sched_dynamic, 1);
    }
    int *dist = allocate(n, sizeof(int));
    int *frontier = allocate(n, sizeof(int));
    int *next = allocate(n, sizeof(int));
    int *local = allocate(n, sizeof(int));
    int *global = allocate(n, sizeof(int));
    List *lists = allocate(threads, sizeof(List));
    int *seq = NULL;
    int *sizes = NULL;
    int *by_rank = NULL;
    if (rank == 0) {
        seq = allocate(n, sizeof(int));
        if (stats) {
            sizes = allocate(n, sizeof(int));
            by_rank = allocate(nprocs, sizeof(int));
        }
    }

    double sequential_time = 0.0;
    if (rank == 0) {
        double begin = MPI_Wtime();
        bfs_sequential(&graph, start, seq, frontier);
        sequential_time = MPI_Wtime() - begin;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int processed;
    int actual_threads;
    double begin = MPI_Wtime();
    int levels = bfs_hybrid(&graph, start, dist, rank, nprocs,
                            frontier, next, local, global, lists, threads,
                            sizes, &processed, &actual_threads);
    double elapsed = MPI_Wtime() - begin;
    double parallel_time = 0.0;
    MPI_Reduce(&elapsed, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long long team_size = actual_threads;
    long long total_threads = 0;
    MPI_Reduce(&team_size, &total_threads, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (stats) {
        MPI_Gather(&processed, 1, MPI_INT, by_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    int mismatches = 0;
    if (rank == 0) {
        int unreachable = 0;
        for (int v = 0; v < n; ++v) {
            unreachable += dist[v] == -1;
            mismatches += seq[v] != dist[v];
        }
        double speedup = parallel_time > 0.0 ? sequential_time / parallel_time : 0.0;
        printf("Вершин: %d, рёбер: %d, записей в CSR: %d\n"
               "Начальная вершина: %d, уровней: %d, недостижимых вершин: %d\n"
               "Несовпадений расстояний: %d\n", n, graph.edge_count,
               graph.offsets[n], start, levels, unreachable, mismatches);
        printf("Проверка: %s\n", mismatches == 0 ? "расстояния совпали" : "есть расхождения");
        printf("Процессов MPI: %d, потоков OpenMP у процесса 0: %d, всего потоков: %lld\n"
               "Время последовательного BFS: %.9f с\n"
               "Время гибридного BFS: %.9f с\n"
               "Ускорение: %.6f, эффективность: %.6f\n",
               nprocs, actual_threads, total_threads, sequential_time,
               parallel_time, speedup, speedup / total_threads);

        if (stats) {
            int max_size = 0;
            printf("Размеры фронтов по уровням:");
            for (int i = 0; i < levels; ++i) {
                printf(" %d", sizes[i]);
                if (sizes[i] > max_size) {
                    max_size = sizes[i];
                }
            }
            printf("\nМаксимальный размер фронта: %d\nОбработано вершин по рангам:", max_size);
            for (int i = 0; i < nprocs; ++i) {
                printf(" %d", by_rank[i]);
            }
            putchar('\n');
        }
    }
    MPI_Bcast(&mismatches, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int t = 0; t < threads; ++t) {
        free(lists[t].items);
    }
    free(lists);
    free(by_rank);
    free(sizes);
    free(seq);
    free(global);
    free(local);
    free(next);
    free(frontier);
    free(dist);
    free(graph.neighbors);
    free(graph.offsets);
    MPI_Finalize();
    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

