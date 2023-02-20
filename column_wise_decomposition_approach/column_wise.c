#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Matrix multiplication (C = AB) using a column-wise decomposition approach.
 * The main complexity of any parallel application is figure out how to assign chunks to processes.
 * In the following code the work is divided depending on the number of C columns.
 * For instance, if C has 4 columns and the program is run with 2 processes, master and worker (with ranks 0 and 1),
 * master will compute columns 0 and 2, worker columns 1 and 3.
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
void calculate_product(int process_rank, int process_size, int a[M][N], int b[N][K], int c[M][K]) {

    for (int i = 0; i < M; i++) {
        for (int j = process_rank; j < K; j = j + process_size) {
            for (int k = 0; k < N; k++) {
                c[i][j] = c[i][j] + a[i][k]*b[k][j];
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

    /* Matrices and variables definition. */
    static int a[M][N], b[N][K], c[M][K];
    int size, rank, i, j;
    double my_time, max_time, min_time, avg_time;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    initialize_matrices(a, b, c);

    /* Barrier used to synchronize every process before "true computation" starts */
    MPI_Barrier(MPI_COMM_WORLD);
    my_time = MPI_Wtime();

    /* Matrix multiplication core */
    calculate_product(rank, size, a, b, c);

    /* Workers send their results to the master */
    if (rank != 0) {
        for (i = rank; i < M; i = i + size) {
            for (j = 0; j < K; j++) {
                MPI_Send(&c[i][j], 1, MPI_INT, 0, i * K + j, MPI_COMM_WORLD);
            }
        }
    }

    /* Master receives the results from workers */
    if (rank == 0) {
        for (int process = 1; process < size; process++) {
            for (i = process; i < M; i = i + size) {
                for (j = 0; j < K; j++) {
                    MPI_Recv(&c[i][j], 1, MPI_INT, process, i * K + j, MPI_COMM_WORLD, &status);
                }
            }
        }
    }

    /* my_time contains the time spent for "true computation" */
    my_time = MPI_Wtime() - my_time;

    /* The min, max and avg computation time are computed and made available to the master */
    MPI_Reduce(&my_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Master prints the result matrix and computation time */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        avg_time = avg_time / size;
        printf("\n");
        print_results(c, min_time, max_time, avg_time);
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d computation time: %f\n", rank, my_time);

    MPI_Finalize();
    return 0;
}
