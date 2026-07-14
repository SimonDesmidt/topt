#include "fem_solver.h"
#include <suitesparse/umfpack.h>

static size_t binomial_coefficient(size_t n, size_t k) {
    if (k > n - k) k = n - k;
    size_t result = 1;
    for (size_t i = 1; i <= k; i++) result = result * (n - k + i) / i;
    return result;
}

static double integer_power(double x, int exponent) {
    double result = 1.0;
    for (int i = 0; i < exponent; i++) result *= x;
    return result;
}

static void accumulate_recursive(int start, int depth, int p, int tuple[], const double density_element[], size_t num_elements, double average[]) {
    if (depth == p) {
        for (size_t element = 0; element < num_elements; element++) {
            double term = 1.0;
            for (int i = 0; i < p; i++) term *= density_element[4 * element + tuple[i]];
            average[element] += term;
        }
        return;
    }

    for (int value = start; value < 4; value++) {
        tuple[depth] = value;
        accumulate_recursive(value, depth + 1, p, tuple, density_element, num_elements, average);
    }
}

static int accumulate_combinations(const double density_element[], size_t num_elements, int p, double average[]) {
    if (p < 0) return 0;
    for (size_t element = 0; element < num_elements; element++) average[element] = 0.0;

    if (p == 0) {
        for (size_t element = 0; element < num_elements; element++) average[element] = 1.0;
        return 1;
    }

    int *tuple = malloc((size_t)p * sizeof(*tuple));
    if (tuple == NULL) return 0;
    accumulate_recursive(0, 0, p, tuple, density_element, num_elements, average);
    free(tuple);
    return 1;
}

int barycentric_integral(const double density_element[], size_t num_elements, int p, double average[]) {
    if (density_element == NULL || average == NULL || p < 0) return 0;
    if (!accumulate_combinations(density_element, num_elements, p, average)) return 0;

    double denominator = (double)binomial_coefficient((size_t)p + 3, 3);
    for (size_t element = 0; element < num_elements; element++) average[element] /= denominator;
    return 1;
}

static void accumulate_derivative_recursive(int start, int depth, int p, int tuple[], const double density_element[], size_t num_elements, double derivative[]) {
    if (depth == p) {
        int multiplicities[4] = {0, 0, 0, 0};
        for (int i = 0; i < p; i++) multiplicities[tuple[i]]++;

        for (int local_node = 0; local_node < 4; local_node++) {
            if (multiplicities[local_node] == 0) continue;

            for (size_t element = 0; element < num_elements; element++) {
                double term = (double)multiplicities[local_node];

                for (int variable = 0; variable < 4; variable++) {
                    int exponent = multiplicities[variable] - (variable == local_node ? 1 : 0);
                    if (exponent > 0) term *= integer_power(density_element[4 * element + variable], exponent);
                }

                derivative[4 * element + local_node] += term;
            }
        }
        return;
    }

    for (int value = start; value < 4; value++) {
        tuple[depth] = value;
        accumulate_derivative_recursive(value, depth + 1, p, tuple, density_element, num_elements, derivative);
    }
}

int barycentric_integral_derivative(const double density_element[], size_t num_elements, int p, double derivative[]) {
    if (density_element == NULL || derivative == NULL || p < 0) return 0;
    memset(derivative, 0, 4 * num_elements * sizeof(*derivative));
    if (p == 0) return 1;

    int *tuple = malloc((size_t)p * sizeof(*tuple));
    if (tuple == NULL) return 0;
    accumulate_derivative_recursive(0, 0, p, tuple, density_element, num_elements, derivative);
    free(tuple);

    double denominator = (double)binomial_coefficient((size_t)p + 3, 3);
    for (size_t i = 0; i < 4 * num_elements; i++) derivative[i] /= denominator;
    return 1;
}

static int sparse_entry_compare(const void *a, const void *b) {
    const SparseEntry *entry_a = a;
    const SparseEntry *entry_b = b;
    if (entry_a->row < entry_b->row) return -1;
    if (entry_a->row > entry_b->row) return 1;
    if (entry_a->col < entry_b->col) return -1;
    if (entry_a->col > entry_b->col) return 1;
    return 0;
}

static int assemble_global_stiffness(const Mesh *mesh, const double element_young[], CsrMatrix *K) {
    const int num_dofs = 3 * mesh->num_nodes;
    const size_t raw_nnz = (size_t)mesh->num_elements * 144;
    SparseEntry *entries = malloc(raw_nnz * sizeof(*entries));
    if (entries == NULL) return 0;

    size_t cursor = 0;
    for (int element = 0; element < mesh->num_elements; element++) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                entries[cursor].row = mesh->edofMat[element][i];
                entries[cursor].col = mesh->edofMat[element][j];
                entries[cursor].value = element_young[element] * mesh->Ke[element][i][j];
                cursor++;
            }
        }
    }

    qsort(entries, cursor, sizeof(*entries), sparse_entry_compare);

    size_t unique_nnz = 0;
    for (size_t k = 0; k < cursor; k++) {
        if (unique_nnz > 0 && entries[unique_nnz - 1].row == entries[k].row && entries[unique_nnz - 1].col == entries[k].col) {
            entries[unique_nnz - 1].value += entries[k].value;
        } else {
            entries[unique_nnz++] = entries[k];
        }
    }

    K->rows = num_dofs;
    K->cols = num_dofs;
    K->nnz = (int)unique_nnz;
    K->row_ptr = calloc((size_t)num_dofs + 1, sizeof(*K->row_ptr));
    K->col_idx = malloc(unique_nnz * sizeof(*K->col_idx));
    K->val = malloc(unique_nnz * sizeof(*K->val));

    if (K->row_ptr == NULL || K->col_idx == NULL || K->val == NULL) {
        free(entries);
        free(K->row_ptr);
        free(K->col_idx);
        free(K->val);
        memset(K, 0, sizeof(*K));
        return 0;
    }

    for (size_t k = 0; k < unique_nnz; k++) K->row_ptr[entries[k].row + 1]++;
    for (int row = 0; row < num_dofs; row++) K->row_ptr[row + 1] += K->row_ptr[row];

    for (size_t k = 0; k < unique_nnz; k++) {
        K->col_idx[k] = entries[k].col;
        K->val[k] = entries[k].value;
    }

    free(entries);
    return 1;
}

static int build_density_elements(const Mesh *mesh, const double density[], double **density_element_out) {
    size_t count = 4 * (size_t)mesh->num_elements;
    double *density_element = malloc(count * sizeof(*density_element));
    if (density_element == NULL) return 0;

    for (int element = 0; element < mesh->num_elements; element++) {
        for (int local_node = 0; local_node < 4; local_node++) {
            int global_node = mesh->elements[element][local_node];
            density_element[4 * (size_t)element + local_node] = density[global_node];
        }
    }

    *density_element_out = density_element;
    return 1;
}

static int assemble_rhs(const Mesh *mesh, const double density_element[], double rhs[]) {
    static const double mass_pattern[4][4] = {
        {2.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0},
        {1.0 / 20.0, 2.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0},
        {1.0 / 20.0, 1.0 / 20.0, 2.0 / 20.0, 1.0 / 20.0},
        {1.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0, 2.0 / 20.0}
    };

    const int num_dofs = 3 * mesh->num_nodes;
    memset(rhs, 0, (size_t)num_dofs * sizeof(*rhs));

    for (int element = 0; element < mesh->num_elements; element++) {
        for (int local_node = 0; local_node < 4; local_node++) {
            double weighted_density = 0.0;
            for (int variable = 0; variable < 4; variable++) weighted_density += density_element[4 * (size_t)element + variable] * mass_pattern[variable][local_node];
            double local_gravity = -rho_matter * gravity * mesh->volumes[element] * weighted_density;
            int global_node = mesh->elements[element][local_node];
            rhs[3 * global_node + 2] += local_gravity;
        }
    }

    for (int condition_id = 0; condition_id < mesh->num_neumann; condition_id++) {
        const NeumannCondition *condition = &mesh->neumann[condition_id];
        if (condition->node < 0 || condition->node >= mesh->num_nodes) return 0;
        rhs[3 * condition->node] += condition->force[0];
        rhs[3 * condition->node + 1] += condition->force[1];
        rhs[3 * condition->node + 2] += condition->force[2];
    }

    return 1;
}

static int collect_dirichlet_data(const Mesh *mesh, unsigned char fixed[], double fixed_values[]) {
    const int num_dofs = 3 * mesh->num_nodes;
    memset(fixed, 0, (size_t)num_dofs * sizeof(*fixed));
    memset(fixed_values, 0, (size_t)num_dofs * sizeof(*fixed_values));

    for (int condition_id = 0; condition_id < mesh->num_dirichlet; condition_id++) {
        const DirichletCondition *condition = &mesh->dirichlet[condition_id];
        if (condition->node < 0 || condition->node >= mesh->num_nodes) return 0;

        for (int component = 0; component < 3; component++) {
            if (!condition->constrained[component]) continue;
            int dof = 3 * condition->node + component;
            if (fixed[dof] && fabs(fixed_values[dof] - condition->value[component]) > 1e-12) return 0;
            fixed[dof] = 1;
            fixed_values[dof] = condition->value[component];
        }
    }

    return 1;
}

static int solve_reduced_system(const CsrMatrix *K, const double rhs[], const unsigned char fixed[], const double fixed_values[], double u[]) {
    const int num_dofs = K->rows;
    int *global_to_free = malloc((size_t)num_dofs * sizeof(*global_to_free));
    int *free_to_global = malloc((size_t)num_dofs * sizeof(*free_to_global));
    if (global_to_free == NULL || free_to_global == NULL) {
        free(global_to_free);
        free(free_to_global);
        return 0;
    }

    int num_free = 0;
    for (int dof = 0; dof < num_dofs; dof++) {
        if (fixed[dof]) {
            global_to_free[dof] = -1;
            u[dof] = fixed_values[dof];
        } else {
            global_to_free[dof] = num_free;
            free_to_global[num_free++] = dof;
        }
    }

    if (num_free == 0) {
        free(global_to_free);
        free(free_to_global);
        return 1;
    }

    size_t reduced_nnz = 0;
    for (int free_row = 0; free_row < num_free; free_row++) {
        int global_row = free_to_global[free_row];
        for (int k = K->row_ptr[global_row]; k < K->row_ptr[global_row + 1]; k++) {
            if (global_to_free[K->col_idx[k]] >= 0) reduced_nnz++;
        }
    }

    int *triplet_rows = malloc(reduced_nnz * sizeof(*triplet_rows));
    int *triplet_cols = malloc(reduced_nnz * sizeof(*triplet_cols));
    double *triplet_values = malloc(reduced_nnz * sizeof(*triplet_values));
    double *rhs_free = malloc((size_t)num_free * sizeof(*rhs_free));
    double *solution_free = calloc((size_t)num_free, sizeof(*solution_free));

    if (triplet_rows == NULL || triplet_cols == NULL || triplet_values == NULL || rhs_free == NULL || solution_free == NULL) {
        free(global_to_free); free(free_to_global); free(triplet_rows); free(triplet_cols); free(triplet_values); free(rhs_free); free(solution_free);
        return 0;
    }

    size_t cursor = 0;
    for (int free_row = 0; free_row < num_free; free_row++) {
        int global_row = free_to_global[free_row];
        rhs_free[free_row] = rhs[global_row];

        for (int k = K->row_ptr[global_row]; k < K->row_ptr[global_row + 1]; k++) {
            int global_col = K->col_idx[k];
            double value = K->val[k];

            if (fixed[global_col]) {
                rhs_free[free_row] -= value * fixed_values[global_col];
            } else {
                triplet_rows[cursor] = free_row;
                triplet_cols[cursor] = global_to_free[global_col];
                triplet_values[cursor] = value;
                cursor++;
            }
        }
    }

    int *column_ptr = malloc(((size_t)num_free + 1) * sizeof(*column_ptr));
    int *row_idx = malloc(reduced_nnz * sizeof(*row_idx));
    double *values = malloc(reduced_nnz * sizeof(*values));

    if (column_ptr == NULL || row_idx == NULL || values == NULL) {
        free(global_to_free); free(free_to_global); free(triplet_rows); free(triplet_cols); free(triplet_values); free(rhs_free); free(solution_free); free(column_ptr); free(row_idx); free(values);
        return 0;
    }

    int status = umfpack_di_triplet_to_col(num_free, num_free, (int)reduced_nnz, triplet_rows, triplet_cols, triplet_values, column_ptr, row_idx, values, NULL);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_triplet_to_col failed with status %d\n", status);
        goto cleanup_failure;
    }

    void *symbolic = NULL;
    void *numeric = NULL;
    status = umfpack_di_symbolic(num_free, num_free, column_ptr, row_idx, values, &symbolic, NULL, NULL);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_symbolic failed with status %d\n", status);
        goto cleanup_failure;
    }

    status = umfpack_di_numeric(column_ptr, row_idx, values, symbolic, &numeric, NULL, NULL);
    umfpack_di_free_symbolic(&symbolic);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_numeric failed with status %d\n", status);
        goto cleanup_failure;
    }

    status = umfpack_di_solve(UMFPACK_A, column_ptr, row_idx, values, solution_free, rhs_free, numeric, NULL, NULL);
    umfpack_di_free_numeric(&numeric);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_solve failed with status %d\n", status);
        goto cleanup_failure;
    }

    for (int free_dof = 0; free_dof < num_free; free_dof++) u[free_to_global[free_dof]] = solution_free[free_dof];

    free(global_to_free); free(free_to_global); free(triplet_rows); free(triplet_cols); free(triplet_values); free(rhs_free); free(solution_free); free(column_ptr); free(row_idx); free(values);
    return 1;

cleanup_failure:
    free(global_to_free); free(free_to_global); free(triplet_rows); free(triplet_cols); free(triplet_values); free(rhs_free); free(solution_free); free(column_ptr); free(row_idx); free(values);
    return 0;
}

int fem_solver(const Mesh *mesh, const double density[], int p, FemResult *result) {
    if (mesh == NULL || density == NULL || result == NULL || p < 0) return 0;
    memset(result, 0, sizeof(*result));

    const size_t num_elements = (size_t)mesh->num_elements;
    const int num_dofs = 3 * mesh->num_nodes;
    double *density_element = NULL;
    double *rho_p_average = malloc(num_elements * sizeof(*rho_p_average));
    double *element_young = malloc(num_elements * sizeof(*element_young));
    unsigned char *fixed = malloc((size_t)num_dofs * sizeof(*fixed));
    double *fixed_values = malloc((size_t)num_dofs * sizeof(*fixed_values));

    result->u = calloc((size_t)num_dofs, sizeof(*result->u));
    result->rhs = malloc((size_t)num_dofs * sizeof(*result->rhs));
    result->num_dofs = num_dofs;

    if (rho_p_average == NULL || element_young == NULL || fixed == NULL || fixed_values == NULL || result->u == NULL || result->rhs == NULL) goto failure;
    if (!build_density_elements(mesh, density, &density_element)) goto failure;
    if (!barycentric_integral(density_element, num_elements, p, rho_p_average)) goto failure;

    for (size_t element = 0; element < num_elements; element++) element_young[element] = E_void + (E_matter - E_void) * rho_p_average[element];
    if (!assemble_global_stiffness(mesh, element_young, &result->K)) goto failure;
    if (!assemble_rhs(mesh, density_element, result->rhs)) goto failure;
    if (!collect_dirichlet_data(mesh, fixed, fixed_values)) goto failure;
    if (!solve_reduced_system(&result->K, result->rhs, fixed, fixed_values, result->u)) goto failure;

    free(density_element); free(rho_p_average); free(element_young); free(fixed); free(fixed_values);
    return 1;

failure:
    free(density_element); free(rho_p_average); free(element_young); free(fixed); free(fixed_values);
    fem_result_free(result);
    return 0;
}

void fem_result_free(FemResult *result) {
    if (result == NULL) return;
    free(result->u);
    free(result->rhs);
    free(result->K.row_ptr);
    free(result->K.col_idx);
    free(result->K.val);
    memset(result, 0, sizeof(*result));
}


int main(int argc, char **argv) {

    printf("\nTests for barycentric integral\n");
    const size_t numEl = 2;
    const int p = 3;

    double density_element[] = {
        0.2, 0.4, 0.7, 1.0,
        0.1, 0.3, 0.5, 0.8};

    double average[numEl];

    if (!barycentric_integral(density_element, numEl, p, average)) {
        return 1;
    }
    printf("[");
    for (size_t element = 0; element < numEl; element++) {
        printf("%.5f ", average[element]);
    }
    printf("]\n\n");

    printf("\nTests for barycentric integral derivative\n");
    double derivative[4*numEl];
    if (!barycentric_integral_derivative(density_element, numEl, p, derivative)){
        return 1;
    }
    
    for (size_t element = 0; element < numEl; element++){
        printf("[");
        for (int node=0; node<4; node++){
            printf("%.4f ", derivative[4*element+node]);
        }
        printf("]\n");
    }
    printf("\n");

    int ierr = 0;
    gmshInitialize(argc, argv, 1, &ierr);

    if (ierr != 0) {
        fprintf(stderr, "Could not initialize Gmsh.\n");
        return EXIT_FAILURE;
    }

    Mesh mesh = {0};

    if (create_mbb_mesh(3.0, 1.0, 0.1, -4e6*9.81, &mesh) != 0) {
        gmshFinalize(&ierr);
        return EXIT_FAILURE;
    }

    double *density = malloc((size_t)mesh.num_nodes * sizeof(*density));

    if (density == NULL) {
        mesh_free(&mesh);
        gmshFinalize(&ierr);
        return EXIT_FAILURE;
    }

    for (int node = 0; node < mesh.num_nodes; node++) {
        density[node] = 1.0;
    }

    double expected_gravity = 0.0;

    for (int element = 0; element < mesh.num_elements; element++) {
        double mean_density = 0.0;

        for (int local_node = 0; local_node < 4; local_node++) {
            int global_node = mesh.elements[element][local_node];
            mean_density += density[global_node];
        }

        mean_density *= 0.25;
        expected_gravity -= rho_matter * gravity * mesh.volumes[element] * mean_density;
    }

    FemResult fem = {0};

    if (!fem_solver(&mesh, density, 3, &fem)) {
        fprintf(stderr, "FEM solve failed.\n");
        free(density);
        mesh_free(&mesh);
        gmshFinalize(&ierr);
        return EXIT_FAILURE;
    }

    printf("\nNumber of nodes: %d\n", mesh.num_nodes);
    printf("Number of DOFs: %d\n", fem.num_dofs);
    printf("Number of DOFs per node: %f\n", (double)fem.num_dofs/(double)mesh.num_nodes);
    printf("K nnz: %d\n\n", fem.K.nnz);

    double ux_max = 0;
    double uy_max = 0;
    double uz_max = 0;   
    for (int node = 0; node < mesh.num_nodes; node++) {
        if (fabs(fem.u[3*node]) > ux_max) ux_max = fabs(fem.u[3*node]);
        if (fabs(fem.u[3*node+1]) > uy_max) uy_max = fabs(fem.u[3*node+1]);
        if (fabs(fem.u[3*node+2]) > uz_max) uz_max = fabs(fem.u[3*node+2]);
    }

    printf("max |u_x| = %.3e\n", ux_max);
    printf("max |u_y| = %.3e\n", uy_max);
    printf("max |u_z| = %.3e\n", uz_max);

    printf("Number of Dirichlet BC nodes: %d\n", mesh.num_dirichlet);
    printf("Number of Neumann BC nodes: %d\n", mesh.num_neumann);

    double rhs_x = 0.0;
    double rhs_y = 0.0;
    double rhs_z = 0.0;
    double rhs_norm_squared = 0.0;

    for (int node = 0; node < mesh.num_nodes; node++) {
        rhs_x += fem.rhs[3 * node];
        rhs_y += fem.rhs[3 * node + 1];
        rhs_z += fem.rhs[3 * node + 2];

        rhs_norm_squared += fem.rhs[3 * node] * fem.rhs[3 * node];
        rhs_norm_squared += fem.rhs[3 * node + 1] * fem.rhs[3 * node + 1];
        rhs_norm_squared += fem.rhs[3 * node + 2] * fem.rhs[3 * node + 2];
    }

    printf("\nsum rhs x: %.3e\n", rhs_x);
    printf("sum rhs y: %.3e\n", rhs_y);
    printf("sum rhs z: %.3e\n", rhs_z);
    printf("norm rhs: %.3e\n", sqrt(rhs_norm_squared));

    double residual_norm_squared = 0.0;
    double rhs_norm_squared_free = 0.0;

    for (int row = 0; row < fem.K.rows; row++) {
        int is_fixed = 0;

        for (int condition_id = 0; condition_id < mesh.num_dirichlet; condition_id++) {
            int node = mesh.dirichlet[condition_id].node;
            int component = row % 3;

            if (row / 3 == node && mesh.dirichlet[condition_id].constrained[component]) {
                is_fixed = 1;
                break;
            }
        }

        if (is_fixed) continue;

        double Ku = 0.0;

        for (int k = fem.K.row_ptr[row]; k < fem.K.row_ptr[row + 1]; k++) {
            Ku += fem.K.val[k] * fem.u[fem.K.col_idx[k]];
        }

        double residual = Ku - fem.rhs[row];
        residual_norm_squared += residual * residual;
        rhs_norm_squared_free += fem.rhs[row] * fem.rhs[row];
    }

    double relative_residual = sqrt(residual_norm_squared) / fmax(sqrt(rhs_norm_squared_free), 1e-30);
    printf("\nrelative free-DOF residual = %.3e\n\n", relative_residual);

    double actual_gravity_plus_external = 0.0;

    for (int node = 0; node < mesh.num_nodes; node++) {
        actual_gravity_plus_external += fem.rhs[3 * node + 2];
    }

    double external_force = 0.0;

    for (int condition = 0; condition < mesh.num_neumann; condition++) {
        external_force += mesh.neumann[condition].force[2];
    }

    printf("expected gravity   = %.3e\n", expected_gravity);
    printf("actual gravity     = %.3e\n", actual_gravity_plus_external - external_force);

    double K_max = 0.0;
    double K_diag_max = 0.0;

    for (int row = 0; row < fem.K.rows; row++) {
        for (int k = fem.K.row_ptr[row]; k < fem.K.row_ptr[row + 1]; k++) {
            double value = fabs(fem.K.val[k]);

            if (value > K_max) {
                K_max = value;
            }

            if (fem.K.col_idx[k] == row && value > K_diag_max) {
                K_diag_max = value;
            }
        }
    }

    printf("\nmax |K_ij|  = %.3e\n", K_max);
    printf("max |K_ii|  = %.3e\n", K_diag_max);

    fem_result_free(&fem);
    free(density);
    mesh_free(&mesh);
    gmshFinalize(&ierr);

    return EXIT_SUCCESS;
}