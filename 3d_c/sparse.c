/*
 * Sparse matrix formats in C: COO and CSR.
 * Builds a small 4x4 sparse matrix, stores it both ways,
 * converts COO -> CSR, and does sparse matrix-vector multiply (SpMV).
 *
 * Matrix (4x4), 7 nonzeros:
 *   [10  0 20  0]
 *   [ 0 30  0  0]
 *   [ 0  0 40 50]
 *   [60  0  0 70]
 */

#include <stdio.h>
#include <stdlib.h>

/* ---------------- COO format ----------------
 * Three parallel arrays, one entry per nonzero:
 *   row[k], col[k], val[k]
 * No ordering required. Easy to build, cheap to append to.
 * Bad for arithmetic (no fast row access).
 */
typedef struct {
    int nnz;      /* number of nonzeros */
    int rows;
    int cols;
    int *row;
    int *col;
    double *val;
} CooMatrix;

/* ---------------- CSR format ----------------
 * row_ptr has (rows+1) entries.
 * row_ptr[i] .. row_ptr[i+1]-1 are the indices into col_idx/val
 * for nonzeros in row i.
 * Fast row access, fast SpMV. Harder to build incrementally.
 */
typedef struct {
    int rows;
    int cols;
    int nnz;
    int *row_ptr;   /* size rows + 1 */
    int *col_idx;   /* size nnz */
    double *val;    /* size nnz */
} CsrMatrix;

CooMatrix coo_create(int rows, int cols, int nnz) {
    CooMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.nnz = nnz;
    m.row = malloc(nnz * sizeof(int));
    m.col = malloc(nnz * sizeof(int));
    m.val = malloc(nnz * sizeof(double));
    return m;
}

void coo_free(CooMatrix *m) {
    free(m->row);
    free(m->col);
    free(m->val);
}

void csr_free(CsrMatrix *m) {
    free(m->row_ptr);
    free(m->col_idx);
    free(m->val);
}

/* Convert COO -> CSR. Entries within a row are not sorted by column
 * here; add a sort pass if you need that. Duplicate (row,col) pairs
 * are NOT summed; do that in the counting pass if needed. */
CsrMatrix coo_to_csr(const CooMatrix *coo) {
    CsrMatrix csr;
    csr.rows = coo->rows;
    csr.cols = coo->cols;
    csr.nnz = coo->nnz;
    csr.row_ptr = calloc(coo->rows + 1, sizeof(int));
    csr.col_idx = malloc(coo->nnz * sizeof(int));
    csr.val = malloc(coo->nnz * sizeof(double));

    /* 1. Count nonzeros per row -> row_ptr[i+1] holds count for row i */
    for (int k = 0; k < coo->nnz; k++) {
        csr.row_ptr[coo->row[k] + 1]++;
    }

    /* 2. Prefix sum -> row_ptr[i] becomes start offset of row i */
    for (int i = 0; i < csr.rows; i++) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    /* 3. Scatter entries into col_idx/val using a cursor per row */
    int *cursor = malloc(csr.rows * sizeof(int));
    for (int i = 0; i < csr.rows; i++) {
        cursor[i] = csr.row_ptr[i];
    }

    for (int k = 0; k < coo->nnz; k++) {
        int r = coo->row[k];
        int dest = cursor[r]++;
        csr.col_idx[dest] = coo->col[k];
        csr.val[dest] = coo->val[k];
    }

    free(cursor);
    return csr;
}

/* Dense y = A * x, A given in CSR. This is the operation CSR is built for. */
void csr_spmv(const CsrMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
            sum += A->val[k] * x[A->col_idx[k]];
        }
        y[i] = sum;
    }
}

/* Same operation directly on COO, for comparison.
 * No row structure to exploit: touch every nonzero, scatter-add into y. */
void coo_spmv(const CooMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->rows; i++) y[i] = 0.0;

    for (int k = 0; k < A->nnz; k++) {
        y[A->row[k]] += A->val[k] * x[A->col[k]];
    }
}

void print_vector(const char *label, const double *v, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++) {
        printf("%s%.1f", i ? ", " : "", v[i]);
    }
    printf("]\n");
}

/* Build a random sparse matrix in COO with a target density (0..1).
 * Uses a hash set-free approach: just allow duplicate (row,col) pairs
 * to land in different slots (fine for COO; would need summing for CSR
 * if you want strict correctness on collisions). */
CooMatrix coo_random(int rows, int cols, double density, unsigned int seed) {
    srand(seed);

    int max_nnz = rows * cols;
    int target_nnz = (int)(density * max_nnz);
    if (target_nnz < 1) target_nnz = 1;

    CooMatrix m = coo_create(rows, cols, target_nnz);

    for (int k = 0; k < target_nnz; k++) {
        m.row[k] = rand() % rows;
        m.col[k] = rand() % cols;
        m.val[k] = (double)(rand() % 100 - 50) / 10.0;  /* range -5.0..4.9 */
    }

    return m;
}

void print_csr_summary(const CsrMatrix *A) {
    printf("CSR: %d x %d, %d nonzeros (%.1f%% density)\n",
           A->rows, A->cols, A->nnz,
           100.0 * A->nnz / ((double)A->rows * A->cols));
}

int main(void) {
    printf("=== Fixed example ===\n");
    int rows = 4, cols = 4, nnz = 7;

    CooMatrix coo = coo_create(rows, cols, nnz);
    /* COO entries added in arbitrary order to show COO build is order-free */
    int rows_in[]  = {0, 0, 1, 2, 2, 3, 3};
    int cols_in[]  = {0, 2, 1, 2, 3, 0, 3};
    double vals_in[] = {10, 20, 30, 40, 50, 60, 70};
    for (int k = 0; k < nnz; k++) {
        coo.row[k] = rows_in[k];
        coo.col[k] = cols_in[k];
        coo.val[k] = vals_in[k];
    }

    CsrMatrix csr = coo_to_csr(&coo);

    printf("CSR row_ptr: ");
    for (int i = 0; i <= csr.rows; i++) printf("%d ", csr.row_ptr[i]);
    printf("\n");

    printf("CSR col_idx: ");
    for (int k = 0; k < csr.nnz; k++) printf("%d ", csr.col_idx[k]);
    printf("\n");

    printf("CSR val:     ");
    for (int k = 0; k < csr.nnz; k++) printf("%.0f ", csr.val[k]);
    printf("\n\n");

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y_csr[4], y_coo[4];

    csr_spmv(&csr, x, y_csr);
    coo_spmv(&coo, x, y_coo);

    print_vector("x", x, 4);
    print_vector("y = A*x (CSR)", y_csr, 4);
    print_vector("y = A*x (COO)", y_coo, 4);

    coo_free(&coo);
    csr_free(&csr);

    printf("\n=== Random sparse matrix ===\n");
    int n = 8;
    CooMatrix rand_coo = coo_random(n, n, 0.2, 42);
    CsrMatrix rand_csr = coo_to_csr(&rand_coo);

    print_csr_summary(&rand_csr);

    double x_rand[8];
    for (int i = 0; i < n; i++) x_rand[i] = 1.0;

    double y_rand[8];
    csr_spmv(&rand_csr, x_rand, y_rand);
    print_vector("y = A*ones(n)", y_rand, n);

    coo_free(&rand_coo);
    csr_free(&rand_csr);
    return 0;
}