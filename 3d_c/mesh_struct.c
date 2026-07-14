#include "mesh_struct.h"

static Node vsub(Node a, Node b) { return (Node){a.x - b.x, a.y - b.y, a.z - b.z}; }
static Node vadd(Node a, Node b) { return (Node){a.x + b.x, a.y + b.y, a.z + b.z}; }
static Node vscale(Node a, double s) { return (Node){a.x * s, a.y * s, a.z * s}; }
static double vdot(Node a, Node b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Node vcross(Node a, Node b) {
    return (Node){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static double vnorm(Node a) { return sqrt(vdot(a, a)); }
static Node vnormalize(Node a) {
    double n = vnorm(a);
    if (n <= TOL) return (Node){0, 0, 0};
    return vscale(a, 1.0 / n);
}

/* 3x3 matrix as 9 doubles, row-major. Solve J^T x = b (used for grad_ref @ inv(J)
 * equivalents) via Cramer's rule -- fine at this size, avoids a general solver. */
static double det3(const double J[9]) {
    return J[0] * (J[4] * J[8] - J[5] * J[7])
         - J[1] * (J[3] * J[8] - J[5] * J[6])
         + J[2] * (J[3] * J[7] - J[4] * J[6]);
}

/* Solve J * x = b for x (3x3 system), Cramer's rule. Returns 0 on success. */
static int solve3(const double J[9], const double b[3], double x[3]) {
    double d = det3(J);
    if (fabs(d) <= 1e-12) return -1;

    double Jc[9];
    for (int col = 0; col < 3; col++) {
        memcpy(Jc, J, sizeof(Jc));
        for (int row = 0; row < 3; row++) Jc[row * 3 + col] = b[row];
        x[col] = det3(Jc) / d;
    }
    return 0;
}

static int check_gmsh_error(int ierr, const char *operation) {
    if (ierr == 0) return 0;

    fprintf(stderr, "Gmsh error during %s: code %d\n", operation, ierr);
    return -1;
}

/* Isotropic constitutive matrix C (6x6), matches the Python `C` (unit E). */
void constitutive_matrix(double nu, double C[6][6]) {
    memset(C, 0, sizeof(double) * 36);
    double factor = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu));
    C[0][0] = C[1][1] = C[2][2] = factor * (1.0 - nu);
    C[0][1] = C[0][2] = C[1][0] = C[1][2] = C[2][0] = C[2][1] = factor * nu;
    C[3][3] = C[4][4] = C[5][5] = factor * (1.0 - 2.0 * nu) / 2.0;
}

/* ---------------- shape_functions ----------------
 * B: strain-displacement matrix, shape (6, 12), for a linear tetrahedron. */
void shape_functions(const Node coords[4], double B[6][12]) {
    double J[9] = {
        coords[1].x - coords[0].x, coords[2].x - coords[0].x, coords[3].x - coords[0].x,
        coords[1].y - coords[0].y, coords[2].y - coords[0].y, coords[3].y - coords[0].y,
        coords[1].z - coords[0].z, coords[2].z - coords[0].z, coords[3].z - coords[0].z
    };

    /* grad_ref rows: d/dxi, d/deta, d/dzeta of N0..N3 in reference coords.
     * grads = grad_ref @ inv(J)  <=>  for each shape function, solve J^T g = grad_ref_row */
    double grad_ref[4][3] = {
        {-1.0, -1.0, -1.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  1.0,  0.0},
        { 0.0,  0.0,  1.0}
    };

    double Jt[9] = { J[0], J[3], J[6], J[1], J[4], J[7], J[2], J[5], J[8] }; /* J^T */

    double dNdx[4], dNdy[4], dNdz[4];
    for (int n = 0; n < 4; n++) {
        double g[3];
        if (solve3(Jt, grad_ref[n], g) != 0) {
            fprintf(stderr, "shape_functions: degenerate Jacobian\n");
            memset(g, 0, sizeof(g));
        }
        dNdx[n] = g[0];
        dNdy[n] = g[1];
        dNdz[n] = g[2];
    }

    memset(B, 0, sizeof(double) * 6 * 12);
    for (int n = 0; n < 4; n++) {
        B[0][3 * n]     = dNdx[n];
        B[1][3 * n + 1] = dNdy[n];
        B[2][3 * n + 2] = dNdz[n];
        B[3][3 * n]     = dNdy[n];
        B[3][3 * n + 1] = dNdx[n];
        B[4][3 * n + 1] = dNdz[n];
        B[4][3 * n + 2] = dNdy[n];
        B[5][3 * n]     = dNdz[n];
        B[5][3 * n + 2] = dNdx[n];
    }
}

/* ---------------- volume ---------------- */
static double tet_volume(const Node coords[4]) {
    Node e1 = vsub(coords[1], coords[0]);
    Node e2 = vsub(coords[2], coords[0]);
    Node e3 = vsub(coords[3], coords[0]);
    double J[9] = { e1.x, e2.x, e3.x, e1.y, e2.y, e3.y, e1.z, e2.z, e3.z };
    double v = fabs(det3(J)) / 6.0;
    return v > 1e-12 ? v : 1e-12;
}

/* ---------------- local_filter_system ----------------
 * K_elem[i][j] = integral(grad(N_i) . grad(N_j)) dV
 * M_elem[i][j] = consistent mass matrix, V/20 * [[2,1,1,1],...] */
int local_filter_system(const Node coords[4], double K_elem[4][4], double M_elem[4][4]) {
    Node e1 = vsub(coords[1], coords[0]);
    Node e2 = vsub(coords[2], coords[0]);
    Node e3 = vsub(coords[3], coords[0]);
    double J[9] = { e1.x, e2.x, e3.x, e1.y, e2.y, e3.y, e1.z, e2.z, e3.z };
    double detJ = det3(J);

    if (fabs(detJ) <= 1e-12) return -1;

    double vol = fabs(detJ) / 6.0;

    /* grad_ref here is (3,4) in the Python code: rows are xi/eta/zeta,
     * columns are N0..N3. B_filter = solve(J^T, grad_ref). */
    double grad_ref_34[3][4] = {
        {-1.0, 1.0, 0.0, 0.0},
        {-1.0, 0.0, 1.0, 0.0},
        {-1.0, 0.0, 0.0, 1.0}
    };

    double Jt[9] = { J[0], J[3], J[6], J[1], J[4], J[7], J[2], J[5], J[8] };

    double Bf[3][4]; /* B_filter, shape (3,4) */
    for (int col = 0; col < 4; col++) {
        double b[3] = { grad_ref_34[0][col], grad_ref_34[1][col], grad_ref_34[2][col] };
        double g[3];
        if (solve3(Jt, b, g) != 0) return -1;
        Bf[0][col] = g[0];
        Bf[1][col] = g[1];
        Bf[2][col] = g[2];
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += Bf[k][i] * Bf[k][j];
            K_elem[i][j] = vol * s;
        }
    }

    static const double mass_pattern[4][4] = {
        {2, 1, 1, 1}, {1, 2, 1, 1}, {1, 1, 2, 1}, {1, 1, 1, 2}
    };
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            M_elem[i][j] = vol * mass_pattern[i][j] / 20.0;

    return 0;
}

/* ---------------- COO -> CSR (same pattern as sparse_formats.c) ---------------- */

static int coo_entry_cmp(const void *a, const void *b) {
    const CooEntry *ea = a, *eb = b;
    if (ea->row != eb->row) return ea->row - eb->row;
    return ea->col - eb->col;
}

/* COO -> CSR, summing duplicate (row,col) entries -- matches scipy's
 * coo_matrix.tocsr(), which sums duplicates by default. */
CsrMatrix coo_to_csr(const CooMatrix *coo) {
    CooEntry *entries = malloc(coo->nnz * sizeof(CooEntry));
    for (int k = 0; k < coo->nnz; k++) {
        entries[k].row = coo->row[k];
        entries[k].col = coo->col[k];
        entries[k].val = coo->val[k];
    }
    qsort(entries, coo->nnz, sizeof(CooEntry), coo_entry_cmp);

    /* Merge adjacent duplicates in place, count unique entries. */
    int unique = 0;
    for (int k = 0; k < coo->nnz; k++) {
        if (unique > 0 && entries[unique - 1].row == entries[k].row
                       && entries[unique - 1].col == entries[k].col) {
            entries[unique - 1].val += entries[k].val;
        } else {
            entries[unique++] = entries[k];
        }
    }

    CsrMatrix csr;
    csr.rows = coo->rows; csr.cols = coo->cols; csr.nnz = unique;
    csr.row_ptr = calloc(coo->rows + 1, sizeof(int));
    csr.col_idx = malloc(unique * sizeof(int));
    csr.val = malloc(unique * sizeof(double));

    for (int k = 0; k < unique; k++) csr.row_ptr[entries[k].row + 1]++;
    for (int i = 0; i < csr.rows; i++) csr.row_ptr[i + 1] += csr.row_ptr[i];

    int *cursor = malloc(csr.rows * sizeof(int));
    memcpy(cursor, csr.row_ptr, csr.rows * sizeof(int));
    for (int k = 0; k < unique; k++) {
        int r = entries[k].row;
        int dest = cursor[r]++;
        csr.col_idx[dest] = entries[k].col;
        csr.val[dest] = entries[k].val;
    }
    free(cursor);
    free(entries);
    return csr;
}

void csr_free(CsrMatrix *m) { free(m->row_ptr); free(m->col_idx); free(m->val); }

void mesh_free(Mesh *mesh) {
    if (mesh == NULL) return;

    free(mesh->nodes);
    free(mesh->elements);
    free(mesh->volumes);
    free(mesh->B_matrices);
    free(mesh->Ke);
    free(mesh->edofMat);
    free(mesh->iK);
    free(mesh->jK);
    free(mesh->sK_unit);
    free(mesh->boundary_normals);
    free(mesh->boundary_tangent_1);
    free(mesh->boundary_tangent_2);
    free(mesh->node_tags);
    free(mesh->dirichlet);
    free(mesh->neumann);
    free(mesh->left_nodes);
    free(mesh->right_nodes);
    free(mesh->bottom_nodes);
    free(mesh->top_nodes);
    free(mesh->load_nodes);
    free(mesh->lower_right_nodes);
    csr_free(&mesh->stiff_matrix);
    csr_free(&mesh->mass_matrix);

    memset(mesh, 0, sizeof(*mesh));
}

/* ---------------- build_all_element_stiffness_matrices ---------------- */
void build_element_stiffness(Mesh *mesh, const double C[6][6]) {
    mesh->Ke = malloc(mesh->num_elements * sizeof(*mesh->Ke));
    mesh->B_matrices = malloc(mesh->num_elements * sizeof(*mesh->B_matrices));
    mesh->volumes = malloc(mesh->num_elements * sizeof(double));

    for (int e = 0; e < mesh->num_elements; e++) {
        Node coords[4];
        for (int n = 0; n < 4; n++) coords[n] = mesh->nodes[mesh->elements[e][n]];

        double V = tet_volume(coords);
        double B[6][12];
        shape_functions(coords, B);

        memcpy(mesh->B_matrices[e], B, sizeof(B));
        mesh->volumes[e] = V;

        /* Ke = V * B^T @ C @ B */
        double CB[6][12];
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 12; j++) {
                double s = 0.0;
                for (int k = 0; k < 6; k++) s += C[i][k] * B[k][j];
                CB[i][j] = s;
            }

        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 12; j++) {
                double s = 0.0;
                for (int k = 0; k < 6; k++) s += B[k][i] * CB[k][j];
                mesh->Ke[e][i][j] = V * s;
            }
    }
}

/* ---------------- build_edof_matrix ---------------- */
void build_edof_matrix(Mesh *mesh) {
    mesh->edofMat = malloc(mesh->num_elements * sizeof(*mesh->edofMat));
    for (int e = 0; e < mesh->num_elements; e++) {
        for (int n = 0; n < 4; n++) {
            int Node = mesh->elements[e][n];
            mesh->edofMat[e][3 * n]     = 3 * Node;
            mesh->edofMat[e][3 * n + 1] = 3 * Node + 1;
            mesh->edofMat[e][3 * n + 2] = 3 * Node + 2;
        }
    }

    /* iK, jK, sK_unit: COO pattern for global stiffness assembly (Ke, unit Young). */
    int per_el = 144;
    mesh->iK = malloc(mesh->num_elements * per_el * sizeof(int));
    mesh->jK = malloc(mesh->num_elements * per_el * sizeof(int));
    mesh->sK_unit = malloc(mesh->num_elements * per_el * sizeof(double));

    int idx = 0;
    for (int e = 0; e < mesh->num_elements; e++) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                mesh->iK[idx] = mesh->edofMat[e][i];
                mesh->jK[idx] = mesh->edofMat[e][j];
                mesh->sK_unit[idx] = mesh->Ke[e][i][j];
                idx++;
            }
        }
    }
}

/* ---------------- helmholtz_filter ----------------
 * Assembles K_filter, M_filter as CSR. Filter radius r = max circumradius / 6
 * (matches this doc's version, note: differs from the /2.0 seen elsewhere). */
void helmholtz_filter(Mesh *mesh, CsrMatrix *K_filter, CsrMatrix *M_filter, double *r_out) {
    double r = 0.0;

    for (int e = 0; e < mesh->num_elements; e++) {
        Node p0 = mesh->nodes[mesh->elements[e][0]];
        Node p1 = mesh->nodes[mesh->elements[e][1]];
        Node p2 = mesh->nodes[mesh->elements[e][2]];
        Node p3 = mesh->nodes[mesh->elements[e][3]];

        /* Circumcenter via 2*(p_i - p0) . x = |p_i|^2 - |p0|^2, i=1,2,3 */
        double A[9] = {
            2*(p1.x-p0.x), 2*(p1.y-p0.y), 2*(p1.z-p0.z),
            2*(p2.x-p0.x), 2*(p2.y-p0.y), 2*(p2.z-p0.z),
            2*(p3.x-p0.x), 2*(p3.y-p0.y), 2*(p3.z-p0.z)
        };
        double b[3] = {
            vdot(p1,p1) - vdot(p0,p0),
            vdot(p2,p2) - vdot(p0,p0),
            vdot(p3,p3) - vdot(p0,p0)
        };
        double center[3];
        if (solve3(A, b, center) == 0) {
            Node c = {center[0], center[1], center[2]};
            double circumradius = vnorm(vsub(c, p0));
            if (circumradius > r) r = circumradius;
        }
    }
    r /= 6.0;
    *r_out = r;

    int nel = mesh->num_elements;
    CooMatrix cooK = { mesh->num_nodes, mesh->num_nodes, nel * 16,
                        malloc(nel*16*sizeof(int)), malloc(nel*16*sizeof(int)), malloc(nel*16*sizeof(double)) };
    CooMatrix cooM = { mesh->num_nodes, mesh->num_nodes, nel * 16,
                        malloc(nel*16*sizeof(int)), malloc(nel*16*sizeof(int)), malloc(nel*16*sizeof(double)) };

    int idx = 0;
    for (int e = 0; e < nel; e++) {
        Node coords[4];
        for (int n = 0; n < 4; n++) coords[n] = mesh->nodes[mesh->elements[e][n]];

        double K_elem[4][4], M_elem[4][4];
        if (local_filter_system(coords, K_elem, M_elem) != 0) {
            fprintf(stderr, "helmholtz_filter: degenerate element %d\n", e);
            continue;
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int ni = mesh->elements[e][i];
                int nj = mesh->elements[e][j];
                cooK.row[idx] = ni; cooK.col[idx] = nj; cooK.val[idx] = K_elem[i][j];
                cooM.row[idx] = ni; cooM.col[idx] = nj; cooM.val[idx] = M_elem[i][j];
                idx++;
            }
        }
    }

    cooK.nnz = idx;
    cooM.nnz = idx;

    *K_filter = coo_to_csr(&cooK);
    *M_filter = coo_to_csr(&cooM);

    free(cooK.row); free(cooK.col); free(cooK.val);
    free(cooM.row); free(cooM.col); free(cooM.val);
}

/* ---------------- boundary_nodes_normals_tangents ---------------- */
void boundary_nodes_normals_tangents(Mesh *mesh) {
    mesh->boundary_normals = calloc(mesh->num_nodes, sizeof(Node));
    mesh->boundary_tangent_1 = calloc(mesh->num_nodes, sizeof(Node));
    mesh->boundary_tangent_2 = calloc(mesh->num_nodes, sizeof(Node));

    static const int local_faces[4][4] = {
        {0, 1, 2, 3}, {0, 3, 1, 2}, {0, 2, 3, 1}, {1, 3, 2, 0}
    };

    /* Hash faces by sorted (i,j,k) triple to find faces used by exactly one tet. */
    typedef struct { int key_a, key_b, key_c, a, b, c, opposite, count; } FaceEntry;
    int cap = mesh->num_elements * 4 * 2;
    FaceEntry *table = calloc(cap, sizeof(FaceEntry));
    for (int i = 0; i < cap; i++) table[i].count = 0;

    for (int e = 0; e < mesh->num_elements; e++) {
        for (int f = 0; f < 4; f++) {
            int a = mesh->elements[e][local_faces[f][0]];
            int b = mesh->elements[e][local_faces[f][1]];
            int c = mesh->elements[e][local_faces[f][2]];
            int opp = mesh->elements[e][local_faces[f][3]];

            int sorted[3] = {a, b, c};
            for (int x = 0; x < 3; x++)
                for (int y = x+1; y < 3; y++)
                    if (sorted[y] < sorted[x]) { int t = sorted[x]; sorted[x] = sorted[y]; sorted[y] = t; }

            unsigned long key = ((unsigned long)sorted[0] * 73856093u) ^
                                 ((unsigned long)sorted[1] * 19349663u) ^
                                 ((unsigned long)sorted[2] * 83492791u);
            int slot = (int)(key % (unsigned long)cap);
            while (table[slot].count != 0 &&
                   !(table[slot].key_a == sorted[0] && table[slot].key_b == sorted[1] && table[slot].key_c == sorted[2])) {
                slot = (slot + 1) % cap;
            }
            if (table[slot].count == 0) {
                table[slot].key_a = sorted[0]; table[slot].key_b = sorted[1]; table[slot].key_c = sorted[2];
                table[slot].a = a; table[slot].b = b; table[slot].c = c;
                table[slot].opposite = opp;
                table[slot].count = 1;
            } else {
                table[slot].count++;
            }
        }
    }

    for (int s = 0; s < cap; s++) {
        if (table[s].count != 1) continue;

        int i = table[s].a, j = table[s].b, k = table[s].c, opp = table[s].opposite;
        Node pi = mesh->nodes[i], pj = mesh->nodes[j], pk = mesh->nodes[k], popp = mesh->nodes[opp];

        Node face_normal = vcross(vsub(pj, pi), vsub(pk, pi));
        double nn = vnorm(face_normal);
        if (nn <= TOL) continue;

        Node centroid = vscale(vadd(vadd(pi, pj), pk), 1.0 / 3.0);
        if (vdot(face_normal, vsub(popp, centroid)) > 0.0) face_normal = vscale(face_normal, -1.0);

        mesh->boundary_normals[i] = vadd(mesh->boundary_normals[i], face_normal);
        mesh->boundary_normals[j] = vadd(mesh->boundary_normals[j], face_normal);
        mesh->boundary_normals[k] = vadd(mesh->boundary_normals[k], face_normal);
    }
    free(table);

    for (int n = 0; n < mesh->num_nodes; n++) {
        Node normal = mesh->boundary_normals[n];
        if (vnorm(normal) <= TOL) continue;
        normal = vnormalize(normal);
        mesh->boundary_normals[n] = normal;

        Node reference = (fabs(normal.x) < 0.9) ? (Node){1,0,0} : (Node){0,1,0};
        Node t1 = vnormalize(vcross(normal, reference));
        Node t2 = vnormalize(vcross(normal, t1));
        mesh->boundary_tangent_1[n] = t1;
        mesh->boundary_tangent_2[n] = t2;
    }
}

static int *extract_entity_tags(const int *dim_tags, size_t dim_tags_n, int expected_dim, size_t *num_tags) {
    *num_tags = 0;

    for (size_t i = 0; i + 1 < dim_tags_n; i += 2) {
        if (dim_tags[i] == expected_dim) (*num_tags)++;
    }

    if (*num_tags == 0) return NULL;

    int *tags = malloc(*num_tags * sizeof(*tags));

    if (tags == NULL) {
        *num_tags = 0;
        return NULL;
    }

    size_t count = 0;

    for (size_t i = 0; i + 1 < dim_tags_n; i += 2) {
        if (dim_tags[i] == expected_dim) tags[count++] = dim_tags[i + 1];
    }

    return tags;
}

static int get_surface_tags_in_box(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int **surface_tags, size_t *num_surface_tags) {
    int ierr = 0;
    int *dim_tags = NULL;
    size_t dim_tags_n = 0;

    gmshModelGetEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, &dim_tags, &dim_tags_n, 2, &ierr);

    if (check_gmsh_error(ierr, "gmshModelGetEntitiesInBoundingBox") != 0) {
        gmshFree(dim_tags);
        return -1;
    }

    *surface_tags = extract_entity_tags(dim_tags, dim_tags_n, 2, num_surface_tags);
    gmshFree(dim_tags);

    if (*num_surface_tags == 0 || *surface_tags == NULL) {
        fprintf(stderr, "No surface found in requested bounding box.\n");
        return -1;
    }

    return 0;
}

static int build_tag_to_index(const size_t *node_tags, int num_nodes, int **tag_to_index_out, size_t *max_tag_out) {
    size_t max_tag = 0;

    for (int i = 0; i < num_nodes; i++) {
        if (node_tags[i] > max_tag) max_tag = node_tags[i];
    }

    int *tag_to_index = malloc((max_tag + 1) * sizeof(*tag_to_index));

    if (tag_to_index == NULL) return -1;

    for (size_t tag = 0; tag <= max_tag; tag++) tag_to_index[tag] = -1;
    for (int i = 0; i < num_nodes; i++) tag_to_index[node_tags[i]] = i;

    *tag_to_index_out = tag_to_index;
    *max_tag_out = max_tag;

    return 0;
}

static int physical_group_node_indices(int dim, int physical_tag, const int *tag_to_index, size_t max_tag, int **indices_out, int *num_indices_out) {
    int ierr = 0;
    size_t *gmsh_tags = NULL;
    size_t gmsh_tags_n = 0;
    double *coordinates = NULL;
    size_t coordinates_n = 0;

    gmshModelMeshGetNodesForPhysicalGroup(dim, physical_tag, &gmsh_tags, &gmsh_tags_n, &coordinates, &coordinates_n, &ierr);

    if (check_gmsh_error(ierr, "gmshModelMeshGetNodesForPhysicalGroup") != 0) {
        gmshFree(gmsh_tags);
        gmshFree(coordinates);
        return -1;
    }

    int *indices = malloc(gmsh_tags_n * sizeof(*indices));

    if (indices == NULL) {
        gmshFree(gmsh_tags);
        gmshFree(coordinates);
        return -1;
    }

    int count = 0;

    for (size_t i = 0; i < gmsh_tags_n; i++) {
        size_t tag = gmsh_tags[i];

        if (tag <= max_tag && tag_to_index[tag] >= 0) {
            indices[count++] = tag_to_index[tag];
        }
    }

    gmshFree(gmsh_tags);
    gmshFree(coordinates);

    *indices_out = indices;
    *num_indices_out = count;

    return 0;
}

static int *intersect_node_sets(const int *set_a, int size_a, const int *set_b, int size_b, int num_nodes, int *intersection_size) {
    unsigned char *in_a = calloc((size_t)num_nodes, sizeof(*in_a));
    unsigned char *in_b = calloc((size_t)num_nodes, sizeof(*in_b));

    if (in_a == NULL || in_b == NULL) {
        free(in_a);
        free(in_b);
        *intersection_size = 0;
        return NULL;
    }

    for (int i = 0; i < size_a; i++) in_a[set_a[i]] = 1;
    for (int i = 0; i < size_b; i++) in_b[set_b[i]] = 1;

    int count = 0;

    for (int node_id = 0; node_id < num_nodes; node_id++) if (in_a[node_id] && in_b[node_id]) count++;

    int *intersection = malloc((size_t)count * sizeof(*intersection));

    if (intersection == NULL && count > 0) {
        free(in_a);
        free(in_b);
        *intersection_size = 0;
        return NULL;
    }

    int index = 0;

    for (int node_id = 0; node_id < num_nodes; node_id++) if (in_a[node_id] && in_b[node_id]) intersection[index++] = node_id;

    free(in_a);
    free(in_b);

    *intersection_size = count;
    return intersection;
}

int create_mbb_mesh(double width, double height, double mesh_size, double total_force, Mesh *mesh) {
    if (mesh == NULL) return -1;

    memset(mesh, 0, sizeof(*mesh));

    if (width <= 0.0 || height <= 0.0 || mesh_size <= 0.0) {
        fprintf(stderr, "create_mbb_mesh: dimensions and mesh size must be positive.\n");
        return -1;
    }

    int ierr = 0;

    gmshClear(&ierr);
    if (check_gmsh_error(ierr, "gmshClear") != 0) return -1;

    gmshModelAdd("MBB_3D", &ierr);
    if (check_gmsh_error(ierr, "gmshModelAdd") != 0) return -1;

    gmshOptionSetNumber("General.Terminal", 0.0, &ierr);
    if (check_gmsh_error(ierr, "set General.Terminal") != 0) return -1;

    double depth = height;

    int beam_tag = gmshModelOccAddBox(0.0, 0.0, 0.0, width, depth, height, -1, &ierr);
    if (check_gmsh_error(ierr, "gmshModelOccAddBox") != 0) return -1;

    gmshModelOccSynchronize(&ierr);
    if (check_gmsh_error(ierr, "gmshModelOccSynchronize") != 0) return -1;

    int volume_entities[] = {beam_tag};

    int pbeam = gmshModelAddPhysicalGroup(3, volume_entities, 1, -1, &ierr);
    if (check_gmsh_error(ierr, "add MBB Beam physical group") != 0) return -1;

    gmshModelSetPhysicalName(3, pbeam, "MBB Beam", &ierr);
    if (check_gmsh_error(ierr, "name MBB Beam physical group") != 0) return -1;

    double largest_dimension = fmax(width, fmax(height, depth));
    double eps = fmax(1e-8, 1e-6 * largest_dimension);

    int *left_surfaces = NULL;
    int *right_surfaces = NULL;
    int *bottom_surfaces = NULL;
    int *top_surfaces = NULL;

    size_t num_left_surfaces = 0;
    size_t num_right_surfaces = 0;
    size_t num_bottom_surfaces = 0;
    size_t num_top_surfaces = 0;

    if (get_surface_tags_in_box(-eps, -eps, -eps, eps, depth + eps, height + eps, &left_surfaces, &num_left_surfaces) != 0) goto fail_surfaces;

    if (get_surface_tags_in_box(width - eps, -eps, -eps, width + eps, depth + eps, height + eps, &right_surfaces, &num_right_surfaces) != 0) goto fail_surfaces;

    if (get_surface_tags_in_box(-eps, -eps, -eps, width + eps, depth + eps, eps, &bottom_surfaces, &num_bottom_surfaces) != 0) goto fail_surfaces;

    if (get_surface_tags_in_box(-eps, -eps, height - eps, width + eps, depth + eps, height + eps, &top_surfaces, &num_top_surfaces) != 0) goto fail_surfaces;

    int pleft = gmshModelAddPhysicalGroup(2, left_surfaces, num_left_surfaces, -1, &ierr);
    if (check_gmsh_error(ierr, "add Left physical group") != 0) goto fail_surfaces;

    gmshModelSetPhysicalName(2, pleft, "Left", &ierr);
    if (check_gmsh_error(ierr, "name Left physical group") != 0) goto fail_surfaces;

    int pright = gmshModelAddPhysicalGroup(2, right_surfaces, num_right_surfaces, -1, &ierr);
    if (check_gmsh_error(ierr, "add Right physical group") != 0) goto fail_surfaces;

    gmshModelSetPhysicalName(2, pright, "Right", &ierr);
    if (check_gmsh_error(ierr, "name Right physical group") != 0) goto fail_surfaces;


    int pbottom = gmshModelAddPhysicalGroup(2, bottom_surfaces, num_bottom_surfaces, -1, &ierr);
    if (check_gmsh_error(ierr, "add Bottom physical group") != 0) goto fail_surfaces;

    gmshModelSetPhysicalName(2, pbottom, "Bottom", &ierr);
    if (check_gmsh_error(ierr, "name Bottom physical group") != 0) goto fail_surfaces;


    int ptop = gmshModelAddPhysicalGroup(2, top_surfaces, num_top_surfaces, -1, &ierr);
    if (check_gmsh_error(ierr, "add Top physical group") != 0) goto fail_surfaces;

    gmshModelSetPhysicalName(2, ptop, "Top", &ierr);
    if (check_gmsh_error(ierr, "name Top physical group") != 0) goto fail_surfaces;

    free(left_surfaces);
    free(right_surfaces);
    free(bottom_surfaces);
    free(top_surfaces);

    left_surfaces = NULL;
    right_surfaces = NULL;
    bottom_surfaces = NULL;
    top_surfaces = NULL;

    gmshOptionSetNumber("Mesh.MeshSizeMin", mesh_size, &ierr);
    if (check_gmsh_error(ierr, "set Mesh.MeshSizeMin") != 0) goto fail_surfaces;

    gmshOptionSetNumber("Mesh.MeshSizeMax", mesh_size, &ierr);
    if (check_gmsh_error(ierr, "set Mesh.MeshSizeMax") != 0) goto fail_surfaces;

    gmshOptionSetNumber("Mesh.ElementOrder", 1.0, &ierr);
    if (check_gmsh_error(ierr, "set Mesh.ElementOrder") != 0) goto fail_surfaces;

    gmshOptionSetNumber("Mesh.MeshSizeMin", mesh_size, &ierr);
    gmshOptionSetNumber("Mesh.MeshSizeMax", mesh_size, &ierr);
    gmshOptionSetNumber("Mesh.Algorithm", 6.0, &ierr);
    gmshOptionSetNumber("Mesh.Algorithm3D", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.RandomSeed", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.RandomFactor", 1e-9, &ierr);
    gmshOptionSetNumber("Mesh.RandomFactor3D", 1e-12, &ierr);
    gmshOptionSetNumber("General.NumThreads", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.MaxNumThreads1D", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.MaxNumThreads2D", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.MaxNumThreads3D", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.MeshSizeFromCurvature", 0.0, &ierr);
    gmshOptionSetNumber("Mesh.MeshSizeExtendFromBoundary", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.ElementOrder", 1.0, &ierr);
    gmshOptionSetNumber("Mesh.Reproducible", 1.0, &ierr);

    gmshModelMeshGenerate(3, &ierr);
    if (check_gmsh_error(ierr, "gmshModelMeshGenerate") != 0) goto fail_surfaces;

    /*
     * Get every Node belonging to the volume, including nodes classified
     * on its boundary.
     */
    size_t *gmsh_node_tags = NULL;
    size_t gmsh_node_tags_n = 0;
    double *gmsh_coordinates = NULL;
    size_t gmsh_coordinates_n = 0;
    double *parametric_coordinates = NULL;
    size_t parametric_coordinates_n = 0;

    gmshModelMeshGetNodes(&gmsh_node_tags, &gmsh_node_tags_n, &gmsh_coordinates, &gmsh_coordinates_n, &parametric_coordinates, &parametric_coordinates_n, 3, beam_tag, 1, 0, &ierr);

    if (check_gmsh_error(ierr, "gmshModelMeshGetNodes") != 0) {
        gmshFree(gmsh_node_tags);
        gmshFree(gmsh_coordinates);
        gmshFree(parametric_coordinates);
        goto fail_surfaces;
    }

    if (gmsh_coordinates_n != 3 * gmsh_node_tags_n) {
        fprintf(stderr, "create_mbb_mesh: inconsistent Node coordinate array.\n");
        gmshFree(gmsh_node_tags);
        gmshFree(gmsh_coordinates);
        gmshFree(parametric_coordinates);
        goto fail_surfaces;
    }

    mesh->num_nodes = (int)gmsh_node_tags_n;
    mesh->nodes = malloc(gmsh_node_tags_n * sizeof(*mesh->nodes));
    mesh->node_tags = malloc(gmsh_node_tags_n * sizeof(*mesh->node_tags));

    if (mesh->nodes == NULL || mesh->node_tags == NULL) {
        gmshFree(gmsh_node_tags);
        gmshFree(gmsh_coordinates);
        gmshFree(parametric_coordinates);
        goto fail_surfaces;
    }

    for (int i = 0; i < mesh->num_nodes; i++) {
        mesh->node_tags[i] = gmsh_node_tags[i];

        mesh->nodes[i].x = gmsh_coordinates[3 * i];
        mesh->nodes[i].y = gmsh_coordinates[3 * i + 1];
        mesh->nodes[i].z = gmsh_coordinates[3 * i + 2];
    }

    int *tag_to_index = NULL;
    size_t max_node_tag = 0;

    if (build_tag_to_index(mesh->node_tags, mesh->num_nodes, &tag_to_index, &max_node_tag) != 0) {
        gmshFree(gmsh_node_tags);
        gmshFree(gmsh_coordinates);
        gmshFree(parametric_coordinates);
        goto fail_surfaces;
    }

    gmshFree(gmsh_node_tags);
    gmshFree(gmsh_coordinates);
    gmshFree(parametric_coordinates);

    /*
     * Extract 4-Node first-order tetrahedra.
     */
    int *element_types = NULL;
    size_t element_types_n = 0;

    size_t **element_tags = NULL;
    size_t *element_tags_n = NULL;
    size_t element_tags_nn = 0;

    size_t **element_node_tags = NULL;
    size_t *element_node_tags_n = NULL;
    size_t element_node_tags_nn = 0;

    gmshModelMeshGetElements(&element_types, &element_types_n, &element_tags, &element_tags_n, &element_tags_nn, &element_node_tags, &element_node_tags_n, &element_node_tags_nn, 3, beam_tag, &ierr);

    if (check_gmsh_error(ierr, "gmshModelMeshGetElements") != 0) {
        free(tag_to_index);
        goto fail_elements;
    }

    int tetrahedron_type_index = -1;

    for (size_t type_index = 0; type_index < element_types_n; type_index++) {
        char *element_name = NULL;
        int element_dimension = 0;
        int element_order = 0;
        int num_element_nodes = 0;
        int num_primary_nodes = 0;
        double *local_coordinates = NULL;
        size_t local_coordinates_n = 0;

        gmshModelMeshGetElementProperties(element_types[type_index], &element_name, &element_dimension, &element_order, &num_element_nodes, &local_coordinates, &local_coordinates_n, &num_primary_nodes, &ierr);

        if (check_gmsh_error(ierr, "gmshModelMeshGetElementProperties") != 0) {
            gmshFree(element_name);
            gmshFree(local_coordinates);
            free(tag_to_index);
            goto fail_elements;
        }

        int is_linear_tetrahedron =
            element_dimension == 3 &&
            element_order == 1 &&
            num_element_nodes == 4 &&
            num_primary_nodes == 4;

        gmshFree(element_name);
        gmshFree(local_coordinates);

        if (is_linear_tetrahedron) {
            tetrahedron_type_index = (int)type_index;
            break;
        }
    }

    if (tetrahedron_type_index < 0) {
        fprintf(stderr, "create_mbb_mesh: no 4-Node linear tetrahedra generated.\n");
        free(tag_to_index);
        goto fail_elements;
    }

    size_t connectivity_count = element_node_tags_n[tetrahedron_type_index];

    if (connectivity_count % 4 != 0) {
        fprintf(stderr, "create_mbb_mesh: invalid tetrahedral connectivity length.\n");
        free(tag_to_index);
        goto fail_elements;
    }

    mesh->num_elements = (int)(element_node_tags_n[tetrahedron_type_index] / 4);
    mesh->elements = malloc((size_t)mesh->num_elements * sizeof(*mesh->elements));

    if (mesh->elements == NULL) {
        free(tag_to_index);
        goto fail_elements;
    }

    for (int element_id = 0; element_id < mesh->num_elements; element_id++) {
        for (int local_node = 0; local_node < 4; local_node++) {
            size_t gmsh_tag = element_node_tags[tetrahedron_type_index][4 * element_id + local_node];

            if (gmsh_tag > max_node_tag || tag_to_index[gmsh_tag] < 0) {
                fprintf(stderr, "create_mbb_mesh: unknown Node tag %zu.\n", gmsh_tag);
                free(tag_to_index);
                goto fail_elements;
            }
            mesh->elements[element_id][local_node] = tag_to_index[gmsh_tag];
        }
    }

    /*
     * Convert physical-group Node tags to zero-based local indices.
     */
    if (physical_group_node_indices(2, pleft, tag_to_index, max_node_tag, &mesh->left_nodes, &mesh->num_left_nodes) != 0) {
        free(tag_to_index);
        goto fail_elements;
    }

    if (physical_group_node_indices(2, pright, tag_to_index, max_node_tag, &mesh->right_nodes, &mesh->num_right_nodes) != 0) {
        free(tag_to_index);
        goto fail_elements;
    }

    if (physical_group_node_indices(2, pbottom, tag_to_index, max_node_tag, &mesh->bottom_nodes, &mesh->num_bottom_nodes) != 0) {
        free(tag_to_index);
        goto fail_elements;
    }

    if (physical_group_node_indices(2, ptop, tag_to_index, max_node_tag, &mesh->top_nodes, &mesh->num_top_nodes) != 0) {
        free(tag_to_index);
        goto fail_elements;
    }

    free(tag_to_index);

    mesh->load_nodes = intersect_node_sets(mesh->left_nodes, mesh->num_left_nodes, mesh->top_nodes, mesh->num_top_nodes, mesh->num_nodes, &mesh->num_load_nodes);

    if (mesh->num_load_nodes == 0 || mesh->load_nodes == NULL) {
        fprintf(stderr, "create_mbb_mesh: no nodes found on upper-left load edge.\n");
        goto fail_elements;
    }

    mesh->lower_right_nodes = intersect_node_sets(mesh->right_nodes, mesh->num_right_nodes, mesh->bottom_nodes, mesh->num_bottom_nodes, mesh->num_nodes, &mesh->num_lower_right_nodes);

    if (mesh->num_lower_right_nodes == 0 || mesh->lower_right_nodes == NULL) {
        fprintf(stderr, "create_mbb_mesh: no nodes found on lower-right support edge.\n");
        goto fail_elements;
    }

    /*
     * Release arrays returned by gmshModelMeshGetElements.
     */
    for (size_t i = 0; i < element_tags_nn; i++) gmshFree(element_tags[i]);
    for (size_t i = 0; i < element_node_tags_nn; i++) gmshFree(element_node_tags[i]);

    gmshFree(element_types);
    gmshFree(element_tags);
    gmshFree(element_tags_n);
    gmshFree(element_node_tags);
    gmshFree(element_node_tags_n);

    element_types = NULL;
    element_tags = NULL;
    element_tags_n = NULL;
    element_node_tags = NULL;
    element_node_tags_n = NULL;

    /*
     * Build element volumes, B matrices, stiffness matrices and DOF maps.
     */
    double C[6][6];
    constitutive_matrix(NU, C);
    build_element_stiffness(mesh, C);
    build_edof_matrix(mesh);

    /*
     * Dirichlet conditions:
     *
     * 1. Entire left face:
     *      ux = 0
     *      uy = 0
     *      uz free
     *
     * 2. Entire lower-right edge:
     *      ux = uy = uz = 0
     *
     * The lower-right edge does not overlap the left face for width > 0,
     * so no duplicate merging is needed here.
     */
    mesh->num_dirichlet = mesh->num_left_nodes + mesh->num_lower_right_nodes;
    mesh->dirichlet = calloc((size_t)mesh->num_dirichlet, sizeof(*mesh->dirichlet));
    if (mesh->dirichlet == NULL) goto fail_elements;

    int condition_id = 0;

    for (int i = 0; i < mesh->num_left_nodes; i++) {
        DirichletCondition *condition = &mesh->dirichlet[condition_id++];

        condition->node = mesh->left_nodes[i];

        condition->constrained[0] = 1;
        condition->constrained[1] = 1;
        condition->constrained[2] = 0;

        condition->value[0] = 0.0;
        condition->value[1] = 0.0;
        condition->value[2] = 0.0;
    }

    for (int i = 0; i < mesh->num_lower_right_nodes; i++) {
        DirichletCondition *condition = &mesh->dirichlet[condition_id++];

        condition->node = mesh->lower_right_nodes[i];

        condition->constrained[0] = 1;
        condition->constrained[1] = 1;
        condition->constrained[2] = 1;

        condition->value[0] = 0.0;
        condition->value[1] = 0.0;
        condition->value[2] = 0.0;
    }

    /*
     * Neumann condition:
     *
     * The supplied F is interpreted as the total force on the complete
     * upper-left edge. It is distributed equally between the edge nodes.
     */
    mesh->num_neumann = mesh->num_load_nodes;
    mesh->neumann = calloc((size_t)mesh->num_neumann, sizeof(*mesh->neumann));
    if (mesh->neumann == NULL) goto fail_elements;

    double nodal_force = total_force / (double)mesh->num_load_nodes;

    for (int i = 0; i < mesh->num_load_nodes; i++) {
        mesh->neumann[i].node = mesh->load_nodes[i];
        mesh->neumann[i].force[0] = 0.0;
        mesh->neumann[i].force[1] = 0.0;
        mesh->neumann[i].force[2] = nodal_force;
    }

    /*
     * Helmholtz filter and boundary directions.
     */
    helmholtz_filter(mesh, &mesh->stiff_matrix, &mesh->mass_matrix, &mesh->r);
    boundary_nodes_normals_tangents(mesh);
    
    // gmshWrite("c_mesh.msh", &ierr);
    return 0;

fail_elements:
    if (element_tags != NULL) {
        for (size_t i = 0; i < element_tags_nn; i++) gmshFree(element_tags[i]);
    }

    if (element_node_tags != NULL) {
        for (size_t i = 0; i < element_node_tags_nn; i++) gmshFree(element_node_tags[i]);
    }

    gmshFree(element_types);
    gmshFree(element_tags);
    gmshFree(element_tags_n);
    gmshFree(element_node_tags);
    gmshFree(element_node_tags_n);

fail_surfaces:
    free(left_surfaces);
    free(right_surfaces);
    free(bottom_surfaces);
    free(top_surfaces);
    mesh_free(mesh);

    return -1;
}

/* ---------------- demo / self-test ---------------- */

// int main(int argc, char **argv) {
//     /* Two tetrahedra sharing one face, forming a small bipyramid.
//      * Nodes: 0=(0,0,0) 1=(1,0,0) 2=(0,1,0) 3=(0,0,1) 4=(1,1,1)
//      * Tet A: 0,1,2,3   Tet B: 1,2,3,4   (shared face 1,2,3) */
//     Node nodes[5] = {
//         {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,1}
//     };
//     int elements[2][4] = {
//         {0, 1, 2, 3},
//         {1, 2, 3, 4}
//     };

//     Mesh mesh = {0};
//     mesh.num_nodes = 5;
//     mesh.num_elements = 2;
//     mesh.nodes = nodes;
//     mesh.elements = elements;

//     double C[6][6];
//     constitutive_matrix(NU, C);

//     build_element_stiffness(&mesh, C);
//     build_edof_matrix(&mesh);

//     printf("=== Element volumes ===\n");
//     for (int e = 0; e < mesh.num_elements; e++)
//         printf("  element %d: V = %.6f\n", e, mesh.volumes[e]);

//     printf("\n=== Ke[0] symmetry check ===\n");
//     double max_asym = 0.0;
//     for (int i = 0; i < 12; i++)
//         for (int j = 0; j < 12; j++) {
//             double diff = fabs(mesh.Ke[0][i][j] - mesh.Ke[0][j][i]);
//             if (diff > max_asym) max_asym = diff;
//         }
//     printf("  max |Ke - Ke^T| = %.3e (should be ~0)\n", max_asym);

//     printf("\n=== Global stiffness pattern ===\n");
//     printf("  nnz (COO, with duplicates from shared face) = %d\n", mesh.num_elements * 144);

//     CsrMatrix K_filter, M_filter;
//     double r;
//     helmholtz_filter(&mesh, &K_filter, &M_filter, &r);
//     printf("\n=== Helmholtz filter ===\n");
//     printf("  filter radius r = %.6f\n", r);
//     printf("  K_filter: %d x %d\n", K_filter.rows, K_filter.cols);
//     for (int row=0; row<K_filter.rows; row++){
//         for (int k=K_filter.row_ptr[row]; k<K_filter.row_ptr[row+1]; k++){
//             int col = K_filter.col_idx[k];
//             double value = K_filter.val[k];
//             printf("(%d, %d)    %.3f\n", row, col, value);
//         }
//     }

//     boundary_nodes_normals_tangents(&mesh);
//     printf("\n=== Boundary normals ===\n");
//     for (int n = 0; n < mesh.num_nodes; n++) {
//         Node nrm = mesh.boundary_normals[n];
//         printf("  Node %d: normal = (%.3f, %.3f, %.3f), |n| = %.3f\n",
//                n, nrm.x, nrm.y, nrm.z, vnorm(nrm));
//     }
//     printf("\n\n");

//     csr_free(&K_filter);
//     csr_free(&M_filter);
//     free(mesh.volumes);
//     free(mesh.B_matrices);
//     free(mesh.Ke);
//     free(mesh.edofMat);
//     free(mesh.iK);
//     free(mesh.jK);
//     free(mesh.sK_unit);
//     free(mesh.boundary_normals);
//     free(mesh.boundary_tangent_1);
//     free(mesh.boundary_tangent_2);


//     int ierr = 0;

//     gmshInitialize(argc, argv, 1, &ierr);

//     if (ierr != 0) {
//         fprintf(stderr, "gmshInitialize failed with error %d\n", ierr);
//         return EXIT_FAILURE;
//     }

//     Mesh mesh2 = {0};
//     double width = 3.0; 
//     double height = 1.0;
//     double h = 0.04;

//     if (create_mbb_mesh(width, height, h, -4000.0 * 9.81, &mesh2) != 0) {
//         gmshFinalize(&ierr);
//         return EXIT_FAILURE;
//     }

//     printf("\n=== Mesh with h=%.3f ===\n", h);
//     printf("Nodes: %d\n", mesh2.num_nodes);
//     printf("Elements: %d\n", mesh2.num_elements);
//     printf("Load nodes: %d\n", mesh2.num_neumann);
//     printf("Support nodes: %d\n", mesh2.num_dirichlet);

//     mesh_free(&mesh2);
//     gmshFinalize(&ierr);

//     return EXIT_SUCCESS;
// }