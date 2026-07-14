#ifndef MESH_STRUCT_H
#define MESH_STRUCT_H

#include "gmshc.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define NU 0.3
#define E_matter 210e9
#define E_void 1.0
#define rho_matter 7850.0
#define gravity 9.81
#define TOL 1e-14

typedef struct {double x,y,z;} Node;

typedef struct {
    int node;
    int constrained[3];
    double value[3];
} DirichletCondition;

typedef struct {
    int node;
    double force[3];
} NeumannCondition;

typedef struct { int rows, cols, nnz; int *row, *col; double *val; } CooMatrix;

/* Comparator + payload for sorting COO entries by (row, col). */
typedef struct { int row, col; double val; } CooEntry;

typedef struct {
    int rows;
    int cols;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *val;
} CsrMatrix;

typedef struct {
    int num_nodes;
    int num_elements;

    Node *nodes;
    int (*elements)[4];

    size_t *node_tags;

    double *volumes;
    double (*B_matrices)[6][12];
    double (*Ke)[12][12];
    int (*edofMat)[12];

    int *iK;
    int *jK;
    double *sK_unit;

    Node *boundary_normals;
    Node *boundary_tangent_1;
    Node *boundary_tangent_2;

    DirichletCondition *dirichlet;
    int num_dirichlet;

    NeumannCondition *neumann;
    int num_neumann;

    int *left_nodes;
    int num_left_nodes;

    int *right_nodes;
    int num_right_nodes;

    int *bottom_nodes;
    int num_bottom_nodes;

    int *top_nodes;
    int num_top_nodes;

    int *load_nodes;
    int num_load_nodes;

    int *lower_right_nodes;
    int num_lower_right_nodes;

    CsrMatrix stiff_matrix;
    CsrMatrix mass_matrix;
    double r;
} Mesh;

void constitutive_matrix(double nu, double C[6][6]);
void shape_functions(const Node coords[4], double B[6][12]);
int local_filter_system(const Node coords[4], double K_elem[4][4], double M_elem[4][4]);
CsrMatrix coo_to_csr(const CooMatrix *coo);
void build_element_stiffness(Mesh *mesh, const double C[6][6]);
void build_edof_matrix(Mesh *mesh);
void helmholtz_filter(Mesh *mesh, CsrMatrix *K_filter, CsrMatrix *M_filter, double *r_out);
void boundary_nodes_normals_tangents(Mesh *mesh);

int create_mbb_mesh(double width, double height, double mesh_size, double total_force, Mesh *mesh);
void mesh_free(Mesh *mesh);
void csr_free(CsrMatrix *matrix);

#endif