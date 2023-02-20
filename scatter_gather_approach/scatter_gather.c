#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Matrix multiplication (C = AB) using a scatter-gather decomposition approach.
 * The main complexity of any parallel application is figure out how to assign chunks to processes.
 * In the following code the work is divided using the MPI functions MPI_Scatter and MPI_Gather.
 * MPI_Scatter divides the A matrix between all available processes.
 * Each process will receive a portion of A represented by local_a variable.
 * After scatter, each process performs its dot product calling calculate_product function.
 * Finally, MPI_Gather is used to collect the locally-calculated results back to the C matrix.
 *
 * Matrix dimensions must be divisible by the number of processes.
 *
*/

#define M 3000
#define N 2000
#define K 4000

/* Function that fills a and b with random values, c with zeros */
void initialize_matrices(int a[M][N], int b[N][K], int c[M][K]) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = rand() % 1000;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i][j] = rand() % 1000;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            c[i][j] = 0;
        }
    }
}

/* Function that contains the dot matrix implementation */
void calculate_product(int *a, int *b, int *c, int m, int n, int p) {

    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < p; j++) {
                c[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}

/* Function that prints the result matrix */
void print_results(int c[M][K], double min_time, double max_time, double avg_time) {

    printf("Min computation time: %f\n", min_time);
    printf("Max computation time: %f\n", max_time);
    printf("Avg computation time: %f\n", avg_time);
}

/* Main function */
int main(int argc, char **argv) {

    /* Matrices and variables definition */
    static int a[M][N], b[N][K], c[M][K];
    int size, rank, i, j;
    double my_time, max_time, min_time, avg_time;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    initialize_matrices(a, b, c);

    MPI_Barrier(MPI_COMM_WORLD);

    int rows = M / size;
    if (rows == 0) {

        if (rank == 0) {
            printf("A matrix is too small!\n");
        }

        MPI_Finalize();
        return 0;
    }

    int *local_a = (int *) malloc(rows * M * sizeof(int));
    int *local_c = (int *) malloc(rows * K * sizeof(int));

    /* Barrier used to synchronize every process before "true computation" starts */
    MPI_Barrier(MPI_COMM_WORLD);
    my_time = MPI_Wtime();

    MPI_Scatter(a, rows * M, MPI_INT, local_a, rows * M, MPI_INT, 0, MPI_COMM_WORLD);
    calculate_product(local_a, *b, local_c, rows, N, K);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(local_c, rows * K, MPI_INT, c, rows * K, MPI_INT, 0, MPI_COMM_WORLD);

    /* my_time contains the time spent for "true computation" */
    my_time = MPI_Wtime() - my_time;

    /* The min, max and avg computation time are computed and made available to the master */
    MPI_Reduce(&my_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Master prints the result matrix and computation time */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n");
        avg_time = avg_time / size;
        print_results(c, min_time, max_time, avg_time);
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d computation time: %f\n", rank, my_time);

    free(local_a);
    free(local_c);

    MPI_Finalize();
    return 0;
}