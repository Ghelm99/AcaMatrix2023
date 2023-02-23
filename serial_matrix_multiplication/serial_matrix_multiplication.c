#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
void calculate_product(int a[M][N], int b[N][K], int c[M][K]) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            c[i][j] = 0;

            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

/* Function that prints the result matrix */
void print_result(int c[M][K], double t_start, double t_finish) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("c[%d][%d]:%d\n", i, j, c[i][j]);
        }
    }

    printf("\n");
    printf("Result computed in %f seconds. \n", (t_finish - t_start) / CLOCKS_PER_SEC);
}

/* Main function */
int main(int argc, char **argv) {

    static int a[M][N], b[N][K], c[M][K];
    clock_t t_start, t_end;

    initialize_matrices(a, b, c);

    t_start = clock();
    calculate_product(a, b, c);
    t_end = clock();

    print_result(c, (double) t_start, (double) t_end);
    return 0;
}
