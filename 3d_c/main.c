#include <stdio.h>
#include <stdlib.h>
#include "gmshc.h"
#include "mesh_struct.h"
#include "fem_solver.h"

int main(int argc, char **argv) {
    int ierr = 0;
    gmshInitialize(argc, argv, 1, &ierr);

    if (ierr != 0) {
        fprintf(stderr, "Could not initialize Gmsh.\n");
        return EXIT_FAILURE;
    }

    Mesh mesh = {0};

    if (create_mbb_mesh(3.0, 1.0, 0.1, -4000.0 * 9.81, &mesh) != 0) {
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

    FemResult fem = {0};

    if (!fem_solver(&mesh, density, 3, &fem)) {
        fprintf(stderr, "FEM solve failed.\n");
        free(density);
        mesh_free(&mesh);
        gmshFinalize(&ierr);
        return EXIT_FAILURE;
    }

    printf("Number of DOFs: %d\n", fem.num_dofs);
    printf("K nnz: %d\n", fem.K.nnz);

    for (int node = 0; node < mesh.num_nodes; node++) {
        printf(
            "node %d: u = (% .6e, % .6e, % .6e)\n",
            node,
            fem.u[3 * node],
            fem.u[3 * node + 1],
            fem.u[3 * node + 2]
        );
    }

    fem_result_free(&fem);
    free(density);
    mesh_free(&mesh);
    gmshFinalize(&ierr);

    return EXIT_SUCCESS;
}