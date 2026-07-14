#ifndef FEM_SOLVER_H
#define FEM_SOLVER_H
#include "mesh_struct.h"

typedef struct {
    int num_dofs;
    double *u;
    double *rhs;
    CsrMatrix K;
} FemResult;

typedef struct {
    int row;
    int col;
    double value;
} SparseEntry;

int barycentric_integral(const double density_element[], size_t num_elements, int p, double average[]);
int barycentric_integral_derivative(const double density_element[], size_t num_elements, int p, double derivative[]);
int fem_solver(const Mesh *mesh, const double density[], int p, FemResult *result);
void fem_result_free(FemResult *result);

#endif
