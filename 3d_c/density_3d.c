#include "fem_solver.h"

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

    for (size_t node = 0; node < num_nodes; node++) {
        gradient[node] = 0.0;
    }

    /*
     * density_element[e, a] = density[mesh.elements[e][a]]
     */
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

    /*
     * unit_energy[e] = ue^T Ke ue
     */
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
        {2.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0},
        {1.0 / 20.0, 2.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0},
        {1.0 / 20.0, 1.0 / 20.0, 2.0 / 20.0, 1.0 / 20.0},
        {1.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0, 2.0 / 20.0}
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

