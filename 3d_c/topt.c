#include "fem_solver.h"
#include "topt.h"
#include <errno.h>
#include <suitesparse/umfpack.h>

static void csr_matvec(const CsrMatrix *A, const double x[], double y[]) {
    for (int row = 0; row < A->rows; row++) {
        double value = 0.0;
        for (int k = A->row_ptr[row]; k < A->row_ptr[row + 1]; k++) value += A->val[k] * x[A->col_idx[k]];
        y[row] = value;
    }
}

static void csr_transpose_matvec(const CsrMatrix *A, const double x[], double y[]) {
    memset(y, 0, (size_t)A->cols * sizeof(*y));

    for (int row = 0; row < A->rows; row++) {
        for (int k = A->row_ptr[row]; k < A->row_ptr[row + 1]; k++) {
            int col = A->col_idx[k];
            y[col] += A->val[k] * x[row];
        }
    }
}

static int matrix_entry_compare(const void *a, const void *b) {
    const MatrixEntry *entry_a = a;
    const MatrixEntry *entry_b = b;

    if (entry_a->row < entry_b->row) return -1;
    if (entry_a->row > entry_b->row) return 1;
    if (entry_a->col < entry_b->col) return -1;
    if (entry_a->col > entry_b->col) return 1;

    return 0;
}

static int csr_linear_combination(const CsrMatrix *A, double alpha, const CsrMatrix *B, double beta, CsrMatrix *C) {
    if (A == NULL || B == NULL || C == NULL) return 0;
    if (A->rows != B->rows || A->cols != B->cols) return 0;

    memset(C, 0, sizeof(*C));

    size_t capacity = (size_t)A->nnz + (size_t)B->nnz;
    MatrixEntry *entries = malloc(capacity * sizeof(*entries));
    if (entries == NULL) return 0;

    size_t cursor = 0;
    for (int row = 0; row < A->rows; row++) {
        for (int k = A->row_ptr[row]; k < A->row_ptr[row + 1]; k++) {
            entries[cursor].row = row;
            entries[cursor].col = A->col_idx[k];
            entries[cursor].value = alpha * A->val[k];
            cursor++;
        }
    }

    for (int row = 0; row < B->rows; row++) {
        for (int k = B->row_ptr[row]; k < B->row_ptr[row + 1]; k++) {
            entries[cursor].row = row;
            entries[cursor].col = B->col_idx[k];
            entries[cursor].value = beta * B->val[k];
            cursor++;
        }
    }

    qsort(entries, cursor, sizeof(*entries), matrix_entry_compare);

    size_t unique_nnz = 0;
    for (size_t k = 0; k < cursor; k++) {
        if (unique_nnz > 0 && entries[unique_nnz - 1].row == entries[k].row && entries[unique_nnz - 1].col == entries[k].col) entries[unique_nnz - 1].value += entries[k].value; 
        else entries[unique_nnz++] = entries[k];
    }

    C->rows = A->rows;
    C->cols = A->cols;
    C->nnz = (int)unique_nnz;

    C->row_ptr = calloc((size_t)C->rows + 1, sizeof(*C->row_ptr));
    C->col_idx = malloc(unique_nnz * sizeof(*C->col_idx));
    C->val = malloc(unique_nnz * sizeof(*C->val));

    if (C->row_ptr == NULL || C->col_idx == NULL || C->val == NULL) {
        free(entries);
        free(C->row_ptr);
        free(C->col_idx);
        free(C->val);
        memset(C, 0, sizeof(*C));
        return 0;
    }

    for (size_t k = 0; k < unique_nnz; k++) C->row_ptr[entries[k].row + 1]++;

    for (int row = 0; row < C->rows; row++) C->row_ptr[row + 1] += C->row_ptr[row];

    for (size_t k = 0; k < unique_nnz; k++) {
        C->col_idx[k] = entries[k].col;
        C->val[k] = entries[k].value;
    }

    free(entries);
    return 1;
}

static void csr_matrix_free(CsrMatrix *A) {
    if (A == NULL) return;

    free(A->row_ptr);
    free(A->col_idx);
    free(A->val);

    memset(A, 0, sizeof(*A));
}

static int umfpack_factorize_csr(const CsrMatrix *A, UmfpackFactor *factor) {
    if (A == NULL || factor == NULL || A->rows != A->cols) return 0;

    memset(factor, 0, sizeof(*factor));

    int *triplet_rows = malloc((size_t)A->nnz * sizeof(*triplet_rows));
    int *triplet_cols = malloc((size_t)A->nnz * sizeof(*triplet_cols));
    double *triplet_values = malloc((size_t)A->nnz * sizeof(*triplet_values));

    factor->column_ptr = malloc(((size_t)A->cols + 1) * sizeof(*factor->column_ptr));
    factor->row_idx = malloc((size_t)A->nnz * sizeof(*factor->row_idx));
    factor->values = malloc((size_t)A->nnz * sizeof(*factor->values));

    if (triplet_rows == NULL || triplet_cols == NULL || triplet_values == NULL || factor->column_ptr == NULL || factor->row_idx == NULL || factor->values == NULL) {
        free(triplet_rows);
        free(triplet_cols);
        free(triplet_values);
        free(factor->column_ptr);
        free(factor->row_idx);
        free(factor->values);
        memset(factor, 0, sizeof(*factor));
        return 0;
    }

    int cursor = 0;
    for (int row = 0; row < A->rows; row++) {
        for (int k = A->row_ptr[row]; k < A->row_ptr[row + 1]; k++) {
            triplet_rows[cursor] = row;
            triplet_cols[cursor] = A->col_idx[k];
            triplet_values[cursor] = A->val[k];
            cursor++;
        }
    }

    int status = umfpack_di_triplet_to_col(A->rows, A->cols, A->nnz, triplet_rows, triplet_cols, triplet_values, factor->column_ptr, factor->row_idx, factor->values, NULL);
    free(triplet_rows);
    free(triplet_cols);
    free(triplet_values);

    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_triplet_to_col failed: %d\n", status);
        return 0;
    }
    void *symbolic = NULL;

    status = umfpack_di_symbolic(A->rows, A->cols, factor->column_ptr, factor->row_idx, factor->values, &symbolic, NULL, NULL);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_symbolic failed: %d\n", status);
        return 0;
    }

    status = umfpack_di_numeric(factor->column_ptr, factor->row_idx, factor->values, symbolic, &factor->numeric, NULL, NULL);
    umfpack_di_free_symbolic(&symbolic);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_numeric failed: %d\n", status);
        return 0;
    }
    factor->n = A->rows;
    return 1;
}

static int umfpack_factor_solve(const UmfpackFactor *factor, const double rhs[], double solution[]) {
    if (factor == NULL || rhs == NULL || solution == NULL || factor->numeric == NULL) return 0;

    int status = umfpack_di_solve(UMFPACK_A, factor->column_ptr, factor->row_idx, factor->values, solution, rhs, factor->numeric, NULL, NULL);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "umfpack_di_solve failed: %d\n", status);
        return 0;
    }
    return 1;
}

static int umfpack_factor_solve_transpose(const UmfpackFactor *factor, const double rhs[], double solution[]) {
    if (factor == NULL || rhs == NULL || solution == NULL || factor->numeric == NULL) return 0;

    int status = umfpack_di_solve(UMFPACK_At, factor->column_ptr, factor->row_idx, factor->values, solution, rhs, factor->numeric, NULL, NULL);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "UMFPACK transpose solve failed: %d\n", status);
        return 0;
    }
    return 1;
}

static void umfpack_factor_free(UmfpackFactor *factor) {
    if (factor == NULL) return;
    if (factor->numeric != NULL) umfpack_di_free_numeric(&factor->numeric);

    free(factor->column_ptr);
    free(factor->row_idx);
    free(factor->values);

    memset(factor, 0, sizeof(*factor));
}

static double vector_dot(const double a[], const double b[], int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) result += a[i] * b[i];
    return result;
}

static void clip_vector(double x[], int n, double minimum, double maximum) {
    for (int i = 0; i < n; i++) {
        if (x[i] < minimum) x[i] = minimum;
        if (x[i] > maximum) x[i] = maximum;
    }
}

int dcompliance_drho(const Mesh *mesh, const double density[], const double u[], int p, double gradient[]) {
    const size_t num_elements = (size_t)mesh->num_elements;
    const size_t num_nodes = (size_t)mesh->num_nodes;

    double *density_element = malloc(4 * num_elements * sizeof(*density_element));
    double *daverage_drho = malloc(4 * num_elements * sizeof(*daverage_drho));
    double *unit_energy = malloc(num_elements * sizeof(*unit_energy));

    if (density_element == NULL || daverage_drho == NULL || unit_energy == NULL) {
        free(density_element);
        free(daverage_drho);
        free(unit_energy);
        return 0;
    }

    for (size_t node = 0; node < num_nodes; node++) gradient[node] = 0.0;

    
    // density_element[e, a] = density[mesh.elements[e][a]]
    for (size_t element = 0; element < num_elements; element++) {
        for (int local_node = 0; local_node < 4; local_node++) {
            int global_node = mesh->elements[element][local_node];
            density_element[4 * element + local_node] = density[global_node];
        }
    }

    if (!barycentric_integral_derivative(density_element, num_elements, p, daverage_drho)) {
        free(density_element);
        free(daverage_drho);
        free(unit_energy);
        return 0;
    }

    // unit_energy[e] = ue^T Ke ue
    for (size_t element = 0; element < num_elements; element++) {
        double energy = 0.0;

        for (int i = 0; i < 12; i++) {
            int global_dof_i = mesh->edofMat[element][i];
            double ui = u[global_dof_i];

            for (int j = 0; j < 12; j++) {
                int global_dof_j = mesh->edofMat[element][j];
                double uj = u[global_dof_j];

                energy += ui * mesh->Ke[element][i][j] * uj;
            }
        }
        unit_energy[element] = energy;
    }

    /*
     * Stiffness contribution:
     *
     * -(E_matter - E_void)
     * * (ue^T Ke ue)
     * * d(rho_p_average)/d(rho_a)
     */
    for (size_t element = 0; element < num_elements; element++) {
        for (int local_node = 0; local_node < 4; local_node++) {
            int global_node = mesh->elements[element][local_node];

            double local_gradient = -(E_matter - E_void) * unit_energy[element] * daverage_drho[4 * element + local_node];

            gradient[global_node] += local_gradient;
        }
    }

    static const double mass_pattern[4][4] = {
        {0.10, 0.05, 0.05, 0.05},
        {0.05, 0.10, 0.05, 0.05},
        {0.05, 0.05, 0.10, 0.05},
        {0.05, 0.05, 0.05, 0.10}
    };

    /*
     * Load contribution:
     *
     * local_load_gradient[e, a]
     * = -rho_matter * gravity * V_e
     *   * sum_b uz[e, b] * mass_pattern[b][a]
     *
     * gradient += 2 * local_load_gradient
     */
    for (size_t element = 0; element < num_elements; element++) {
        double uz_element[4];

        for (int local_node = 0; local_node < 4; local_node++) {
            int global_node = mesh->elements[element][local_node];
            uz_element[local_node] = u[3 * global_node + 2];
        }

        for (int local_density_node = 0; local_density_node < 4; local_density_node++) {
            double weighted_uz = 0.0;

            for (int displacement_node = 0; displacement_node < 4; displacement_node++) {
                weighted_uz += uz_element[displacement_node] * mass_pattern[displacement_node][local_density_node];
            }

            double local_load_gradient = -rho_matter * gravity * mesh->volumes[element] * weighted_uz;

            int global_node = mesh->elements[element][local_density_node];
            gradient[global_node] += 2.0 * local_load_gradient;
        }
    }

    free(density_element);
    free(daverage_drho);
    free(unit_energy);

    return 1;
}

static int oc_update(const double x[], const double dc[], const double dv[], int num_nodes, double volume_target, double xnew[]) {
    if (x == NULL || dc == NULL || dv == NULL || xnew == NULL || num_nodes <= 0) return 0;

    double l1 = 0.0;
    double l2 = 1e9;
    const double move = 0.2;

    int biter = 0;
    const int max_biter = 200;

    while ((l2 - l1) / (l2 + 1e-12) > 1e-3 && biter < max_biter) {
        double lmid = 0.5 * (l1 + l2);
        double current_volume = 0.0;

        for (int node = 0; node < num_nodes; node++) {
            double denominator = dv[node] * lmid;
            if (denominator < 1e-30) denominator = 1e-30;

            double ratio = -dc[node] / denominator;
            if (ratio < 1e-12) ratio = 1e-12;

            double candidate = x[node] * sqrt(ratio);
            if (candidate > x[node] + move) candidate = x[node] + move;
            if (candidate > 1.0) candidate = 1.0;
            if (candidate < x[node] - move) candidate = x[node] - move;
            if (candidate < 0) candidate = 0;
            xnew[node] = candidate;
            current_volume += dv[node] * xnew[node];
        }
        if (current_volume > volume_target) l1 = lmid;
        else                                l2 = lmid;

        biter++;
    }
    return 1;
}

int density_approach(const Mesh *mesh, double volfrac, const double *radius, int p, int max_iter, const double init_design[], int log, DensityResult *result) {
    if (mesh == NULL || result == NULL || mesh->num_nodes <= 0 || mesh->num_elements <= 0 || volfrac <= 0.0 || volfrac > 1.0 || p < 0 || max_iter < 0) return 0;

    memset(result, 0, sizeof(*result));

    const int num_nodes = mesh->num_nodes;
    const int num_dofs = 3 * num_nodes;
    const double threshold = 2e-3;
    const double filter_radius = radius == NULL ? mesh->r : *radius;

    double total_volume = 0.0;

    for (int element = 0; element < mesh->num_elements; element++) total_volume += mesh->volumes[element];

    double *design = malloc((size_t)num_nodes * sizeof(*design));
    double *old_design = malloc((size_t)num_nodes * sizeof(*old_design));
    double *new_design = malloc((size_t)num_nodes * sizeof(*new_design));
    double *density = malloc((size_t)num_nodes * sizeof(*density));

    double *filter_rhs = malloc((size_t)num_nodes * sizeof(*filter_rhs));
    double *dV_drho = calloc((size_t)num_nodes, sizeof(*dV_drho));
    double *volume_adjoint = malloc((size_t)num_nodes * sizeof(*volume_adjoint));
    double *dV_ddesign = malloc((size_t)num_nodes * sizeof(*dV_ddesign));

    double *dc_drho = malloc((size_t)num_nodes * sizeof(*dc_drho));
    double *compliance_adjoint = malloc((size_t)num_nodes * sizeof(*compliance_adjoint));
    double *dc_ddesign = malloc((size_t)num_nodes * sizeof(*dc_ddesign));

    CsrMatrix H_operator = {0};
    UmfpackFactor H_factor = {0};
    FemResult fem = {0};

    if (design == NULL || old_design == NULL || new_design == NULL || density == NULL || filter_rhs == NULL || dV_drho == NULL || volume_adjoint == NULL || dV_ddesign == NULL || dc_drho == NULL || compliance_adjoint == NULL || dc_ddesign == NULL) goto failure;

    if (init_design == NULL) {
        for (int node = 0; node < num_nodes; node++) design[node] = volfrac;
    } 
    else memcpy(design, init_design, (size_t)num_nodes * sizeof(*design));

    memcpy(density, design, (size_t)num_nodes * sizeof(*density));

    if (!csr_linear_combination(&mesh->stiff_matrix, filter_radius * filter_radius, &mesh->mass_matrix, 1.0, &H_operator)) {
        fprintf(stderr, "Could not construct Helmholtz operator.\n");
        goto failure;
    }

    if (!umfpack_factorize_csr(&H_operator, &H_factor)) {
        fprintf(stderr, "Could not factorize Helmholtz operator.\n");
        goto failure;
    }

    // dV/drho for linear tetrahedral density interpolation.
    for (int element = 0; element < mesh->num_elements; element++) {
        double contribution = mesh->volumes[element] / 4.0;

        for (int local_node = 0; local_node < 4; local_node++) {
            int global_node = mesh->elements[element][local_node];
            dV_drho[global_node] += contribution;
        }
    }

    // volume_adjoint = H^{-T} dV/drho
    if (!umfpack_factor_solve_transpose(&H_factor, dV_drho, volume_adjoint)) goto failure;

    // dV/ddesign = M^T volume_adjoint
    csr_transpose_matvec(&mesh->mass_matrix, volume_adjoint, dV_ddesign);

    if (log && max_iter > 0) {
        result->compliances = malloc((size_t)max_iter * sizeof(*result->compliances));
        result->volume_fractions = malloc((size_t)max_iter * sizeof(*result->volume_fractions));
        result->variations = malloc((size_t)max_iter * sizeof(*result->variations));
        result->density_history = malloc((size_t)max_iter * (size_t)num_nodes * sizeof(*result->density_history));

        if (result->compliances == NULL || result->volume_fractions == NULL || result->variations == NULL || result->density_history == NULL) goto failure;
    }

    printf("=== 3D topology optimization with Helmholtz PDE filter ===\n\n");
    printf("Elements: %d, Nodes: %d\n", mesh->num_elements, mesh->num_nodes);
    printf("Volume fraction: %.3f, p: %d\n", volfrac, p);

    int iteration = 0;
    double change = 1.0;
    double compliance = NAN;

    while (change > threshold && iteration < max_iter) {
        // density = clip(H^{-1} M design, 0, 1)
        csr_matvec(&mesh->mass_matrix, design, filter_rhs);

        if (!umfpack_factor_solve(&H_factor, filter_rhs, density)) goto failure;

        clip_vector(density, num_nodes, 0.0, 1.0);

        // Structural solve.
        // fem_result_free(&fem);

        if (!fem_solver(mesh, density, p, &fem)) {
            fprintf(stderr, "FEM solve failed at iteration %d.\n", iteration);
            goto failure;
        }

        compliance = vector_dot(fem.u, fem.rhs, num_dofs);
        // dc/drho
        if (!dcompliance_drho(mesh, density, fem.u, p, dc_drho)) {
            fprintf(stderr, "Compliance gradient failed at iteration %d.\n", iteration);
            goto failure;
        }

        // compliance_adjoint = H^{-T} dc/drho
        if (!umfpack_factor_solve_transpose(&H_factor, dc_drho, compliance_adjoint)) goto failure;

        // dc/ddesign = M^T compliance_adjoint
        csr_transpose_matvec(&mesh->mass_matrix, compliance_adjoint, dc_ddesign);

        memcpy(old_design, design, (size_t)num_nodes * sizeof(*old_design));

        if (!oc_update(old_design, dc_ddesign, dV_ddesign, num_nodes, volfrac * total_volume, new_design)) goto failure;

        memcpy(design, new_design, (size_t)num_nodes * sizeof(*design));

        double difference_norm_squared = 0.0;
        double old_design_norm_squared = 0.0;

        for (int node = 0; node < num_nodes; node++) {
            double difference = design[node] - old_design[node];
            difference_norm_squared += difference * difference;
            old_design_norm_squared += old_design[node] * old_design[node];
        }

        change = sqrt(difference_norm_squared) / fmax(sqrt(old_design_norm_squared), 1e-14);

        double physical_volume = 0.0;

        for (int element = 0; element < mesh->num_elements; element++) {
            double element_density = 0.0;

            for (int local_node = 0; local_node < 4; local_node++) {
                int global_node = mesh->elements[element][local_node];
                element_density += density[global_node];
            }

            element_density *= 0.25;
            physical_volume += mesh->volumes[element] * element_density;
        }

        double volume_fraction = physical_volume / total_volume;

        if (log) {
            result->compliances[iteration] = compliance;
            result->volume_fractions[iteration] = volume_fraction;
            result->variations[iteration] = change;

            memcpy(&result->density_history[(size_t)iteration * (size_t)num_nodes], density, (size_t)num_nodes * sizeof(*density));
        }

        iteration++;
        printf("  iteration %4d  |  compliance = %.4e  |  volume = %.4f  |  design variation = %.4f\n", iteration, compliance, volume_fraction, change);
    }

    result->density = malloc((size_t)num_nodes * sizeof(*result->density));
    result->u = malloc((size_t)num_dofs * sizeof(*result->u));
    result->rhs = malloc((size_t)num_dofs * sizeof(*result->rhs));

    if (result->density == NULL || result->u == NULL || result->rhs == NULL) goto failure;

    memcpy(result->density, density, (size_t)num_nodes * sizeof(*result->density));
    memcpy(result->u, fem.u, (size_t)num_dofs * sizeof(*result->u));
    memcpy(result->rhs, fem.rhs, (size_t)num_dofs * sizeof(*result->rhs));

    result->iterations = iteration;
    result->num_nodes = num_nodes;
    result->num_dofs = num_dofs;
    result->history_length = log ? iteration : 0;
    result->compliance = compliance;

    fem_result_free(&fem);
    umfpack_factor_free(&H_factor);
    csr_matrix_free(&H_operator);

    free(design);
    free(old_design);
    free(new_design);
    free(density);
    free(filter_rhs);
    free(dV_drho);
    free(volume_adjoint);
    free(dV_ddesign);
    free(dc_drho);
    free(compliance_adjoint);
    free(dc_ddesign);

    return 1;

failure:
    fem_result_free(&fem);
    umfpack_factor_free(&H_factor);
    csr_matrix_free(&H_operator);

    free(design);
    free(old_design);
    free(new_design);
    free(density);
    free(filter_rhs);
    free(dV_drho);
    free(volume_adjoint);
    free(dV_ddesign);
    free(dc_drho);
    free(compliance_adjoint);
    free(dc_ddesign);

    density_result_free(result);
    return 0;
}

void density_result_free(DensityResult *result) {
    if (result == NULL) return;

    free(result->density);
    free(result->u);
    free(result->rhs);

    free(result->compliances);
    free(result->volume_fractions);
    free(result->variations);
    free(result->density_history);

    memset(result, 0, sizeof(*result));
}

static int write_binary_array(const char *filename, const double values[], size_t count) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        fprintf(stderr, "Could not open '%s': %s\n", filename, strerror(errno));
        return 0;
    }

    size_t written = fwrite(values, sizeof(*values), count, file);

    if (written != count) {
        fprintf(stderr, "Could not completely write '%s'.\n", filename);
        fclose(file);
        return 0;
    }

    if (fclose(file) != 0) {
        fprintf(stderr, "Could not close '%s'.\n", filename);
        return 0;
    }

    return 1;
}

static int save_density_result(const char *output_directory, const Mesh *mesh, const DensityResult *result) {
    if (output_directory == NULL || mesh == NULL || result == NULL) return 0;

    char filename[4096];
    snprintf(filename, sizeof(filename), "%s/metadata.txt", output_directory);
    FILE *metadata = fopen(filename, "w");

    if (metadata == NULL) {
        fprintf(stderr, "Could not open '%s': %s\n", filename, strerror(errno));
        return 0;
    }

    fprintf(metadata, "num_nodes %d\n", result->num_nodes);
    fprintf(metadata, "num_elements %d\n", mesh->num_elements);
    fprintf(metadata, "num_dofs %d\n", result->num_dofs);
    fprintf(metadata, "iterations %d\n", result->iterations);
    fprintf(metadata, "history_length %d\n", result->history_length);
    fprintf(metadata, "compliance %.17g\n", result->compliance);

    if (fclose(metadata) != 0) return 0;

    snprintf(filename, sizeof(filename), "%s/nodes.bin", output_directory);
    if (!write_binary_array(filename, (const double *)mesh->nodes, 3 * (size_t)mesh->num_nodes)) return 0;

    snprintf(filename, sizeof(filename), "%s/elements.bin", output_directory);
    FILE *elements_file = fopen(filename, "wb");

    if (elements_file == NULL) {
        fprintf(stderr, "Could not open '%s': %s\n", filename, strerror(errno));
        return 0;
    }

    size_t element_count = 4 * (size_t)mesh->num_elements;

    if (fwrite(mesh->elements, sizeof(int), element_count, elements_file) != element_count) {
        fclose(elements_file);
        return 0;
    }

    if (fclose(elements_file) != 0) return 0;

    snprintf(filename, sizeof(filename), "%s/density.bin", output_directory);
    if (!write_binary_array(filename, result->density, (size_t)result->num_nodes)) return 0;

    snprintf(filename, sizeof(filename), "%s/u.bin", output_directory);
    if (!write_binary_array(filename, result->u, (size_t)result->num_dofs)) return 0;

    snprintf(filename, sizeof(filename), "%s/rhs.bin", output_directory);
    if (!write_binary_array(filename, result->rhs, (size_t)result->num_dofs)) return 0;

    if (result->history_length > 0) {
        snprintf(filename, sizeof(filename), "%s/compliances.bin", output_directory);
        if (!write_binary_array(filename, result->compliances, (size_t)result->history_length)) return 0;

        snprintf(filename, sizeof(filename), "%s/volume_fractions.bin", output_directory);
        if (!write_binary_array(filename, result->volume_fractions, (size_t)result->history_length)) return 0;

        snprintf(filename, sizeof(filename), "%s/variations.bin", output_directory);
        if (!write_binary_array(filename, result->variations, (size_t)result->history_length)) return 0;

        snprintf(filename, sizeof(filename), "%s/density_history.bin", output_directory);
        size_t history_count = (size_t)result->history_length * (size_t)result->num_nodes;

        if (!write_binary_array(filename, result->density_history, history_count)) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s OUTPUT_DIRECTORY VOLUME_FRACTION\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_directory = argv[1];

    char *end=NULL;
    errno = 0;
    double volfrac = strtod(argv[2], &end);
    if (errno !=0 || end == argv[2] || *end != '\0' || volfrac <= 0.0 || volfrac > 1.0){
        fprintf(stderr, "Invalid volume fraction '%s'; expected a value in (0, 1].\n", argv[2]);
        return EXIT_FAILURE;
    }

    end=NULL;
    errno = 0;
    double meshsize = strtod(argv[3], &end);
    if (errno !=0 || end == argv[3] || *end != '\0' || meshsize <= 0.0){
        fprintf(stderr, "Invalid mesh size'%s'.\n", argv[3]);
        return EXIT_FAILURE;
    }

    int ierr = 0;
    gmshInitialize(argc, argv, 1, &ierr);
    gmshOptionSetNumber("General.Terminal", 0.0, &ierr);

    if (ierr != 0) {
        fprintf(stderr, "Could not initialize Gmsh.\n");
        return EXIT_FAILURE;
    }

    Mesh mesh = {0};

    if (create_mbb_mesh(3.0, 1.0, meshsize, -4e3 * 9.81, &mesh) != 0) {
        gmshFinalize(&ierr);
        return EXIT_FAILURE;
    }

    DensityResult result = {0};

    if (!density_approach(&mesh, volfrac, NULL, 3, 1000, NULL, 1, &result)) {
        fprintf(stderr, "Topology optimization failed.\n");
        mesh_free(&mesh);
        gmshFinalize(&ierr);
        return EXIT_FAILURE;
    }

    if (!save_density_result(output_directory, &mesh, &result)) {
        fprintf(stderr, "Could not save optimization results.\n");
        density_result_free(&result);
        mesh_free(&mesh);
        gmshFinalize(&ierr);
        return EXIT_FAILURE;
    }

    density_result_free(&result);
    mesh_free(&mesh);
    gmshFinalize(&ierr);

    return EXIT_SUCCESS;
}