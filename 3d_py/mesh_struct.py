import gmsh 
import numpy as np 
import scipy.sparse as sp

# Material properties
E_void = 1.0        # Young modulus of void [Pa]
E_matter = 210e9    # Young modulus of steel [Pa]
nu = 0.3            # Poisson ratio [-]
rho_matter = 7850.0 # density of steel [kg/m³]
gravity = 9.81      # gravity constant [m/s²]

# 3D isotropic constitutive matrix
C = (1 / ((1 + nu) * (1 - 2 * nu))) * np.array([
    [1 - nu, nu,     nu,     0,              0,              0],
    [nu,     1 - nu, nu,     0,              0,              0],
    [nu,     nu,     1 - nu, 0,              0,              0],
    [0,      0,      0,      (1 - 2 * nu)/2, 0,              0],
    [0,      0,      0,      0,              (1 - 2 * nu)/2, 0],
    [0,      0,      0,      0,              0,              (1 - 2 * nu)/2]
])

class Mesh:
    def __init__(self):
        self.nodes = None
        self.node_tags = None
        self.elements = None
        self.volumes = None
        self.B_matrices = None
        self.Ke = None
        self.edofMat = None
        self.iK = None
        self.jK = None

        self.dirichlet = []
        self.neumann = []

        self.stiff_matrix = None
        self.mass_matrix = None
        self.r = 0.0

        self.boundary_normals = None
        self.boundary_tangent_1 = None
        self.boundary_tangent_2 = None

def shape_functions(node_coords):
    """
    node_coords: array of shape (4, 3) containing tetrahedron node coordinates.

    Returns:
        B: strain-displacement matrix of shape (6, 12)
    """
    J = np.column_stack((
        node_coords[1] - node_coords[0],
        node_coords[2] - node_coords[0],
        node_coords[3] - node_coords[0]
    ))

    grad_ref = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])

    grads = grad_ref @ np.linalg.inv(J)

    dNdx = grads[:, 0]
    dNdy = grads[:, 1]
    dNdz = grads[:, 2]

    B = np.zeros((6, 12))
    B[0, 0::3] = dNdx
    B[1, 1::3] = dNdy
    B[2, 2::3] = dNdz
    B[3, 0::3] = dNdy
    B[3, 1::3] = dNdx
    B[4, 1::3] = dNdz
    B[4, 2::3] = dNdy
    B[5, 0::3] = dNdz
    B[5, 2::3] = dNdx

    return B

def volume(v):
    """Compute the volume of a tetrahedron given its four vertices."""
    J = np.column_stack((v[1] - v[0], v[2] - v[0], v[3] - v[0]))
    return max(abs(np.linalg.det(J)) / 6.0, 1e-12)

def local_filter_system(nodes_coords):
    """
    Compute the local stiffness and consistent mass matrices for a 4-node
    linear tetrahedral element used in a 3D Helmholtz filter.

    Parameters
    ----------
    nodes_coords : ndarray, shape (4, 3)
        Coordinates of the tetrahedron vertices.

    Returns
    -------
    K_elem : ndarray, shape (4, 4)
        K_ij = integral(grad(N_i) . grad(N_j)) dOmega.
    M_elem : ndarray, shape (4, 4)
        M_ij = integral(N_i * N_j) dOmega.
    """
    J = np.column_stack((nodes_coords[1] - nodes_coords[0], nodes_coords[2] - nodes_coords[0], nodes_coords[3] - nodes_coords[0]))
    detJ = np.linalg.det(J)

    if detJ <= 1e-12: raise ValueError("Degenerate or inverted tetrahedral element.")

    volume = detJ / 6.0

    grad_ref = np.array([
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0]
    ])

    B_filter = np.linalg.solve(J.T, grad_ref)

    K_elem = volume * (B_filter.T @ B_filter)

    M_elem = volume * np.array([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 2.0]
    ]) / 20.0

    return K_elem, M_elem

def helmholtz_filter(mesh):
    """
    Assemble the 3D Helmholtz filter matrices for a linear tetrahedral mesh.

    Filter equation:
        (r**2 * K_filter + M_filter) @ rho = M_filter @ design

    Returns
    -------
    K_filter : scipy.sparse.csr_matrix
        Global Laplacian stiffness matrix.
    M_filter : scipy.sparse.csr_matrix
        Global consistent mass matrix.
    r : float
        Filter radius.
    """
    nnodes = len(mesh.nodes)

    r = 0.0
    for element_nodes in mesh.elements:
        coords = mesh.nodes[element_nodes]
        
        p0 = coords[0]
        A = 2.0 * (coords[1:] - p0)
        b = np.sum(coords[1:]**2, axis=1) - np.dot(p0, p0)

        try:
            center = np.linalg.solve(A, b)
            circumradius = np.linalg.norm(center - p0)
        except np.linalg.LinAlgError:
            circumradius = 0.0

        r = max(r, circumradius)

    r /= 6.0

    iK_filter = []
    jK_filter = []
    sK_filter = []

    iM_filter = []
    jM_filter = []
    sM_filter = []

    for element_nodes in mesh.elements:
        coords = mesh.nodes[element_nodes]
        K_elem, M_elem = local_filter_system(coords)

        for i in range(4):
            for j in range(4):
                iK_filter.append(element_nodes[i])
                jK_filter.append(element_nodes[j])
                sK_filter.append(K_elem[i, j])

                iM_filter.append(element_nodes[i])
                jM_filter.append(element_nodes[j])
                sM_filter.append(M_elem[i, j])

    K_filter = sp.coo_matrix((sK_filter, (iK_filter, jK_filter)), shape=(nnodes, nnodes)).tocsr()
    M_filter = sp.coo_matrix((sM_filter, (iM_filter, jM_filter)), shape=(nnodes, nnodes)).tocsr()

    return K_filter, M_filter, r

def build_all_element_stiffness_matrices(mesh):
    """Precompute all 12 x 12 stiffness matrices for linear tetrahedral elements."""
    KE_all = np.zeros((len(mesh.elements), 12, 12))

    for el_id, element_nodes in enumerate(mesh.elements):
        coords = mesh.nodes[element_nodes]
        V = volume(coords)
        B = shape_functions(coords)

        # K_e = integral(B.T @ C @ B dV)
        # B and C are constant inside a linear tetrahedron.
        KE_all[el_id] = V * (B.T @ C @ B)

    return KE_all

def build_edof_matrix(mesh):
    """Build the element DOF matrix for 4-node tetrahedral elements."""
    nel = len(mesh.elements)
    edofMat = np.zeros((nel, 12), dtype=int)

    for el_idx, element_nodes in enumerate(mesh.elements):
        # DOF ordering:
        # [n0_x, n0_y, n0_z, n1_x, n1_y, n1_z,
        #  n2_x, n2_y, n2_z, n3_x, n3_y, n3_z]
        dofs = []

        for node_id in element_nodes:
            dofs.extend([3 * node_id, 3 * node_id + 1, 3 * node_id + 2])

        edofMat[el_idx, :] = dofs

    return edofMat

def boundary_nodes_normals_tangents(nodes, elements):
    """
    Compute boundary nodes, averaged outward normals, and two tangent vectors
    for a tetrahedral mesh.

    Parameters
    ----------
    nodes : ndarray, shape (nnodes, 3)
        Node coordinates.
    elements : ndarray, shape (nelems, 4)
        Tetrahedral element connectivity.

    Returns
    -------
    boundary_nodes : ndarray
        Indices of boundary nodes.
    normals : ndarray, shape (nnodes, 3)
        Averaged unit outward normal at each boundary node.
    tangent_1 : ndarray, shape (nnodes, 3)
        First unit tangent vector.
    tangent_2 : ndarray, shape (nnodes, 3)
        Second unit tangent vector.
    """
    nodes = np.asarray(nodes, dtype=float)[:, :3]
    elements = np.asarray(elements, dtype=int)

    face_data = {}

    local_faces = (
        (0, 1, 2, 3),
        (0, 3, 1, 2),
        (0, 2, 3, 1),
        (1, 3, 2, 0)
    )

    for tet in elements:
        for a, b, c, opposite in local_faces:
            face = (tet[a], tet[b], tet[c])
            key = tuple(sorted(face))

            if key not in face_data:
                face_data[key] = [1, face, tet[opposite]]
            else:
                face_data[key][0] += 1

    boundary_faces = [(face, opposite) for count, face, opposite in face_data.values() if count == 1]

    normals = np.zeros((len(nodes), 3), dtype=float)
    boundary_mask = np.zeros(len(nodes), dtype=bool)

    for face, opposite_node in boundary_faces:
        i, j, k = face
        pi, pj, pk = nodes[i], nodes[j], nodes[k]
        popposite = nodes[opposite_node]

        face_normal = np.cross(pj - pi, pk - pi)
        normal_norm = np.linalg.norm(face_normal)

        if normal_norm <= 1e-14:
            continue

        face_centroid = (pi + pj + pk) / 3.0

        # The outward normal points away from the tetrahedron's opposite node.
        if np.dot(face_normal, popposite - face_centroid) > 0.0:
            face_normal *= -1.0

        # Keep the magnitude proportional to twice the face area.
        normals[i] += face_normal
        normals[j] += face_normal
        normals[k] += face_normal

        boundary_mask[i] = True
        boundary_mask[j] = True
        boundary_mask[k] = True

    boundary_nodes = np.where(boundary_mask)[0]

    normal_norms = np.linalg.norm(normals[boundary_nodes], axis=1)
    good = normal_norms > 1e-14
    normals[boundary_nodes[good]] /= normal_norms[good, None]

    tangent_1 = np.zeros_like(normals)
    tangent_2 = np.zeros_like(normals)

    for node_id in boundary_nodes:
        normal = normals[node_id]

        if np.linalg.norm(normal) <= 1e-14: continue

        # Choose a reference direction that is not nearly parallel to the normal.
        if abs(normal[0]) < 0.9:
            reference = np.array([1.0, 0.0, 0.0])
        else:
            reference = np.array([0.0, 1.0, 0.0])

        tangent_1[node_id] = np.cross(normal, reference)
        tangent_1[node_id] /= np.linalg.norm(tangent_1[node_id])

        tangent_2[node_id] = np.cross(normal, tangent_1[node_id])
        tangent_2[node_id] /= np.linalg.norm(tangent_2[node_id])

    return normals, tangent_1, tangent_2


def create_mbb_mesh(width, height, mesh_size, F):
    """
    Create a 3D MBB beam with a square depth x height cross-section.

    Geometry:
        x in [0, width]   beam length
        y in [0, depth]   horizontal transverse direction
        z in [0, height]  vertical direction
    """
    already_initialized = gmsh.isInitialized()

    if not already_initialized:
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add("MBB_3D")
    gmsh.option.setNumber("General.Terminal", 0)

    depth = height
    beam = gmsh.model.occ.addBox(0.0, 0.0, 0.0, width, depth, height)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [beam], name="MBB Beam")

    eps = max(1e-8, 1e-6 * max(width, height, depth))

    # Boundary surfaces
    left = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps,
        eps, depth + eps, height + eps, dim=2)

    right = gmsh.model.getEntitiesInBoundingBox(
        width - eps, -eps, -eps,
        width + eps, depth + eps, height + eps, dim=2)

    bottom = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps,
        width + eps, depth + eps, eps, dim=2)

    top = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, height - eps,
        width + eps, depth + eps, height + eps, dim=2)

    pleft = gmsh.model.addPhysicalGroup(2, [tag for _, tag in left], name="Left")
    pright = gmsh.model.addPhysicalGroup(2, [tag for _, tag in right], name="Right")
    pbottom = gmsh.model.addPhysicalGroup(2, [tag for _, tag in bottom], name="Bottom")
    ptop = gmsh.model.addPhysicalGroup(2, [tag for _, tag in top], name="Top")

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 1e-9)
    gmsh.option.setNumber("Mesh.RandomFactor3D", 1e-12)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Reproducible", 1)
    gmsh.model.mesh.generate(3)

    # Get all volume nodes, including boundary nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=3, includeBoundary=True)
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)

    tag2idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
    idx2tag = node_tags.copy()

    # Extract only 4-node linear tetrahedra
    element_types, _, element_node_tags = gmsh.model.mesh.getElements(dim=3)

    tetrahedra_tags = None

    for element_type, connectivity in zip(element_types, element_node_tags):
        _, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)

        if dim == 3 and order == 1 and num_nodes == 4:
            tetrahedra_tags = np.asarray(connectivity, dtype=np.int64).reshape(-1, 4)
            break

    if tetrahedra_tags is None:
        raise RuntimeError("No 4-node linear tetrahedral elements were generated.")

    elements = np.array([[tag2idx[int(tag)] for tag in tetrahedron] for tetrahedron in tetrahedra_tags], dtype=np.int64)

    def physical_group_nodes(dim, physical_tag):
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, physical_tag)
        return np.array([tag2idx[int(tag)] for tag in tags], dtype=np.int64)

    left_nodes = physical_group_nodes(2, pleft)
    right_nodes = physical_group_nodes(2, pright)
    bottom_nodes = physical_group_nodes(2, pbottom)
    top_nodes = physical_group_nodes(2, ptop)

    # Load node at the middle of the upper-left edge
    load_nodes = np.intersect1d(left_nodes, top_nodes)
    
    if len(load_nodes) == 0:
        raise RuntimeError("No nodes found on the upper-left load edge.")

    if not already_initialized:
        gmsh.finalize()

    mesh = Mesh()
    mesh.elements = elements
    mesh.nodes = node_coords
    mesh.node_tags = idx2tag

    nelems = len(mesh.elements)

    mesh.volumes = np.zeros(nelems)
    mesh.B_matrices = np.zeros((nelems, 6, 12))

    for element_id, element_nodes in enumerate(mesh.elements):
        coords = mesh.nodes[element_nodes]
        mesh.B_matrices[element_id] = shape_functions(coords)
        mesh.volumes[element_id] = volume(coords)

    mesh.Ke = build_all_element_stiffness_matrices(mesh)
    mesh.edofMat = build_edof_matrix(mesh)

    # Each tetrahedron has 12 displacement DOFs
    mesh.iK = np.repeat(mesh.edofMat, 12, axis=1).ravel()
    mesh.jK = np.tile(mesh.edofMat, (1, 12)).ravel()

    # 3D displacement components:
    # component 0 = ux
    # component 1 = uy
    # component 2 = uz
    #
    # MBB-like constraints:
    #   ux = 0 on the left face
    #   uy = 0 at the lower-right support
    #   uz = 0 at one node to remove the remaining rigid-body mode
    lower_right_nodes = np.intersect1d(right_nodes, bottom_nodes)
    
    # Left surface: prevent horizontal motion, allow vertical motion.
    # (node_id, ux, uy, uz)
    mesh.dirichlet = [(int(node), 0.0, 0.0, None) for node in left_nodes]

    # Lower-right line: fully fixed.
    mesh.dirichlet.extend((int(node), 0.0, 0.0, 0.0) for node in lower_right_nodes)
    
    nodal_force = F / len(load_nodes)
    mesh.neumann = [(int(node), 0.0, 0.0, nodal_force) for node in load_nodes]

    stiffness, mass, radius = helmholtz_filter(mesh)
    mesh.stiff_matrix = stiffness
    mesh.mass_matrix = mass
    mesh.r = radius

    mesh.boundary_normals, mesh.boundary_tangent_1, mesh.boundary_tangent_2 = boundary_nodes_normals_tangents(mesh.nodes, mesh.elements)

    return mesh


def create_lbracket_mesh(width, height, mesh_size, F, leg_frac=0.4):
    """
    Create a 3D L-bracket: classic topology optimization benchmark.

    Geometry:
        x in [0, width]   horizontal leg length
        y in [0, depth]   out-of-plane thickness (depth = leg thickness)
        z in [0, height]  vertical leg length

    An outer box is cut by a box occupying the top-right region,
    leaving an L shape. The inner corner (x = leg_frac*width,
    z = leg_frac*height) is the classic reentrant-corner stress
    concentration used to test density/level-set switching criteria.

    Fixed: top face of the vertical leg (z = height, x <= leg_frac*width).
    Loaded: bottom edge at the tip of the horizontal leg (x = width, z = 0),
    force F applied in -z.
    """
    already_initialized = gmsh.isInitialized()

    if not already_initialized:
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add("Lbracket_3D")
    gmsh.option.setNumber("General.Terminal", 0)

    depth = leg_frac * height  # leg thickness in y
    outer = gmsh.model.occ.addBox(0.0, 0.0, 0.0, width, depth, height)

    cut_x0 = leg_frac * width
    cut_z0 = leg_frac * height
    pad = max(width, height) * 0.1
    cutter = gmsh.model.occ.addBox(cut_x0, -pad, cut_z0, width - cut_x0 + pad, depth + 2 * pad, height - cut_z0 + pad)

    out_dimtags, _ = gmsh.model.occ.cut([(3, outer)], [(3, cutter)])
    gmsh.model.occ.synchronize()

    bracket = out_dimtags[0][1]
    gmsh.model.addPhysicalGroup(3, [bracket], name="L-bracket")

    eps = max(1e-8, 1e-6 * max(width, height, depth))

    top_leg_top = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, height - eps,
        cut_x0 + eps, depth + eps, height + eps, dim=2)

    tip_faces = gmsh.model.getEntitiesInBoundingBox(
        width - eps, -eps, -eps,
        width + eps, depth + eps, height + eps, dim=2)

    bottom_faces = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps,
        width + eps, depth + eps, eps, dim=2)

    p_fixed = gmsh.model.addPhysicalGroup(2, [tag for _, tag in top_leg_top], name="Fixed")
    p_tip = gmsh.model.addPhysicalGroup(2, [tag for _, tag in tip_faces], name="Tip")
    p_bottom = gmsh.model.addPhysicalGroup(2, [tag for _, tag in bottom_faces], name="Bottom")

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(3)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=3, includeBoundary=True)
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)

    tag2idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
    idx2tag = node_tags.copy()

    element_types, _, element_node_tags = gmsh.model.mesh.getElements(dim=3)

    tetrahedra_tags = None

    for element_type, connectivity in zip(element_types, element_node_tags):
        _, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)

        if dim == 3 and order == 1 and num_nodes == 4:
            tetrahedra_tags = np.asarray(connectivity, dtype=np.int64).reshape(-1, 4)
            break

    if tetrahedra_tags is None:
        raise RuntimeError("No 4-node linear tetrahedral elements were generated.")

    elements = np.array([[tag2idx[int(tag)] for tag in tetrahedron] for tetrahedron in tetrahedra_tags], dtype=np.int64)

    def physical_group_nodes(dim, physical_tag):
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, physical_tag)
        return np.array([tag2idx[int(tag)] for tag in tags], dtype=np.int64)

    fixed_nodes = physical_group_nodes(2, p_fixed)
    tip_nodes = physical_group_nodes(2, p_tip)
    bottom_nodes = physical_group_nodes(2, p_bottom)

    # Load edge: tip face intersected with bottom face (the loaded corner edge).
    load_nodes = np.intersect1d(tip_nodes, bottom_nodes)

    if len(load_nodes) == 0:
        raise RuntimeError("No nodes found on the tip load edge.")

    gmsh.write("python_mesh.msh")
    if not already_initialized:
        gmsh.finalize()

    mesh = Mesh()
    mesh.elements = elements
    mesh.nodes = node_coords
    mesh.node_tags = idx2tag

    nelems = len(mesh.elements)

    mesh.volumes = np.zeros(nelems)
    mesh.B_matrices = np.zeros((nelems, 6, 12))

    for element_id, element_nodes in enumerate(mesh.elements):
        coords = mesh.nodes[element_nodes]
        mesh.B_matrices[element_id] = shape_functions(coords)
        mesh.volumes[element_id] = volume(coords)

    mesh.Ke = build_all_element_stiffness_matrices(mesh)
    mesh.edofMat = build_edof_matrix(mesh)

    mesh.iK = np.repeat(mesh.edofMat, 12, axis=1).ravel()
    mesh.jK = np.tile(mesh.edofMat, (1, 12)).ravel()

    # Fully clamp the top face of the vertical leg.
    mesh.dirichlet = [(int(node), 0.0, 0.0, 0.0) for node in fixed_nodes]

    # Downward tip load, split over the loaded edge nodes.
    nodal_force = F / len(load_nodes)
    mesh.neumann = [(int(node), 0.0, 0.0, -nodal_force) for node in load_nodes]

    stiffness, mass, radius = helmholtz_filter(mesh)
    mesh.stiff_matrix = stiffness
    mesh.mass_matrix = mass
    mesh.r = radius

    mesh.boundary_normals, mesh.boundary_tangent_1, mesh.boundary_tangent_2 = boundary_nodes_normals_tangents(mesh.nodes, mesh.elements)
    return mesh

if __name__ == "__main__":
    nodes = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,1]])
    elements = np.array([[0,1,2,3], [1,2,3,4]])
    mesh = Mesh()
    mesh.nodes = nodes
    mesh.elements = elements 
    print("=== Element volumes ===")
    for i, element in enumerate(mesh.elements):
        print(f"element {i}: V = {volume(mesh.nodes[element]):.6f}")
    # print("\n")
    
    print("=== Ke[0] symmetry check ===")
    Ke = build_all_element_stiffness_matrices(mesh)
    val = 0.0
    for i in range(Ke.shape[0]):
        Keee = Ke[i]
        val = max(np.linalg.norm(Keee-Keee.T, np.inf), val)
    print(f"max |Ke - Ke^T| = {val:3e} should be ~0\n")
    
    print("=== Global stiffness pattern ===")
    print(f"nnz (COO, with duplicates from shared face {len(mesh.elements)*144}\n")
    
    K, M, r = helmholtz_filter(mesh)
    size = K.shape[0]
    nnz = K.getnnz()
    print("=== Helmholtz filter ===")
    print(f"  filter radius r = {r:.6f}")
    print(f"  K_filter: {size}x{size}")
    K = K.tocoo()
    for i in range(len(K.data)):
        print(f"({K.coords[0][i]}, {K.coords[1][i]})     {K.data[i]:.3f}")
    
    print("\n=== Boundary normals ===")
    normals, _, _ = boundary_nodes_normals_tangents(mesh.nodes, mesh.elements)
    for i in range(len(mesh.nodes)):
        print(f"node {i}: normal = ({normals[i][0]:.3f}, {normals[i][1]:.3f}, {normals[i][2]:.3f}), |n| = {np.linalg.norm(normals[i])}")
    print("\n")
    
    h = 0.2
    mesh = create_mbb_mesh(3.0, 1.0, h, -4e6*9.81)
    print(f"\n=== Mesh with h={h:.3f} ===")
    print(f"Nodes: {len(mesh.nodes)}")
    print(f"Node tags: {len(mesh.node_tags)}")
    print(f"Elements: {len(mesh.elements)}")
    print(f"Load nodes: {len(mesh.neumann)}")
    print(f"Support nodes: {len(mesh.dirichlet)}")