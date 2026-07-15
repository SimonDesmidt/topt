#include "mesh_struct.h"
#include "fem_solver.h"

typedef struct {
    int n;
    int *column_ptr;
    int *row_idx;
    double *values;
    void *numeric;
} UmfpackFactor;

typedef struct {
    int iterations;
    int num_nodes;
    int num_dofs;
    int history_length;

    double compliance;
    double *density;
    double *u;
    double *rhs;

    double *compliances;
    double *volume_fractions;
    double *variations;
    double *density_history;
} DensityResult;

int density_approach(
    const Mesh *mesh,
    double volfrac,
    const double *radius,
    int p,
    int max_iter,
    const double init_design[],
    int log,
    DensityResult *result
);

typedef struct {
    int row;
    int col;
    double value;
} MatrixEntry;

void density_result_free(DensityResult *result);