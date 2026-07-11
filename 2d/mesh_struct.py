import gmsh 
import numpy as np 
import scipy.sparse as sp

# Material properties
E_void = 1        # Young modulus of void [Pa]
E_matter = 210e9  # Young modulus of steel [Pa]
nu = 0.3          # Poisson ratio [-]
rho_matter = 7850 # density of steel [kg/m³]
gravity = 9.81    # gravity constant [m/s²]

# Plane stress constitutive matrix
C = (1 / (1 - nu**2)) * np.array([
    [1,  nu, 0],
    [nu, 1,  0],
    [0,  0,  (1-nu)/2]
])

class Mesh:
    def __init__(self):
        self.nodes = []        # node_id -> (x, y)
        self.elements = []     # list of node ID lists (nElems x 3)
        self.areas = []        # area of each element
        self.B_matrices = []   # matrix of shape function derivatives 
        self.Ke = []           # local stiffness matrices without density, jac* BTCB
        self.edofMat = []
        self.iK = []
        self.jK = []
        self.dirichlet = []    # list of fixed DOF (node_id, ux, uy) tuples
        self.neumann = []      # list of (node_id, fx, fy) tuples
        self.stiff_matrix = [] # stiffness matrix Kf for Helmholtz : r² Kf + M 
        self.mass_matrix = []  # mass matrix M
        self.r = 0             # radius of the filter 
        self.borders = []      # node indices of the border
        self.corners = []      # node indices of the corners
        self.node_tags = []    # index to gmsh tag array
        self.neigh = []        # indices of neighbour nodes for all nodes
        self.neighstart = []   # indices of start for the neighbours of each node to store in a 1D array
        self.boundary_normals = []
        self.boundary_tangents = []

def shape_functions(node_coords):
    """ 
    node_coords is the array of the node coordinates of the element  
    Returns:
        the matrix of the derivatives of shape functions on the element defined by the input coordinates 
    """
    x = node_coords[:,0]
    y = node_coords[:,1]
    dxdxi = np.array([[x[1]-x[0], x[2]-x[0]], [y[1]-y[0], y[2]-y[0]]]).T
    dxidx = 1/(2*area(node_coords)) * np.array([[dxdxi[1,1], -dxdxi[0,1]], [-dxdxi[1,0], dxdxi[0,0]]])

    grad_1 = dxidx @ -np.ones(2)
    grad_2 = dxidx @ np.array([1,0])
    grad_3 = dxidx @ np.array([0,1])
    B = np.array([[grad_1[0], 0, grad_2[0], 0, grad_3[0], 0], [0, grad_1[1], 0, grad_2[1], 0, grad_3[1]], [grad_1[1], grad_1[0], grad_2[1], grad_2[0], grad_3[1], grad_3[0]]])
    return B 

def area(v):
    """ Compute the area of a triangle given its vertices """
    return max(0.5 * abs(v[0,0]*(v[1,1]-v[2,1]) + v[1,0]*(v[2,1]-v[0,1]) + v[2,0]*(v[0,1]-v[1,1])), 1e-12)

def local_filter_system(nodes_coords):
    """
    Compute element stiffness and mass matrices for Helmholtz filter
    K_e = int grad(phi_i) . grad(phi_j) dOmega
    M_e = int phi_i * phi_j dOmega
    """
    x = nodes_coords[:,0]
    y = nodes_coords[:,1]
    dxdxi = np.array([[x[1]-x[0], x[2]-x[0]], [y[1]-y[0], y[2]-y[0]]]).T
    dxidx = 1/(2*area(nodes_coords)) * np.array([[dxdxi[1,1], -dxdxi[0,1]], [-dxdxi[1,0], dxdxi[0,0]]])

    B_filter = dxidx @ np.array([[-1,1,0], [-1,0,1]])
    surface = area(nodes_coords)
    # Stiffness matrix K_e = jac * B^T * B
    K_elem = surface * (B_filter.T @ B_filter)
    
    # Mass matrix M_e for linear elements (using 1-point quadrature at centroid)
    # For linear triangles: int phi_i * phi_j = area/12 if i!=j, area/6 if i==j
    M_elem = 2*surface * np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ]) / 24.0
    
    return K_elem, M_elem

def helmholtz_filter(mesh):
    """
    Build Helmholtz filter matrices
    Filter equation: r^2 . K . rho + M . rho = M . design
    Where rho is the filtered density and design the design variable 

    Returns the matrices H = (r^2 * K + M) and M
    """
    nnodes = len(mesh.nodes)
    
    r = 0
    for id, elem in enumerate(mesh.elements):
        p1 = mesh.nodes[elem[0]]
        p2 = mesh.nodes[elem[1]]
        p3 = mesh.nodes[elem[2]]

        area = mesh.areas[id]
        a = np.linalg.norm(np.array(p2)-np.array(p3))
        b = np.linalg.norm(np.array(p3)-np.array(p1))
        c = np.linalg.norm(np.array(p1)-np.array(p2))
        if area != 0.0:
            r = max(r, a*b*c/(4*area))
    # r /= np.sqrt(12)
    r /= 2

    # Build global K and M matrices for the filter
    iK_filter = []
    jK_filter = []
    sK_filter = []
    iM_filter = []
    jM_filter = []
    sM_filter = []
    
    for element_nodes in mesh.elements:
        coords = np.array([mesh.nodes[nid] for nid in element_nodes])
        K_elem, M_elem = local_filter_system(coords)
        
        # Assemble into global matrices
        for i in range(3):
            for j in range(3):
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
    """Precompute all element stiffness matrices"""
    KE_all = np.zeros((len(mesh.elements), 6, 6))
    
    for el_id, element_nodes in enumerate(mesh.elements):
        coords = np.array([mesh.nodes[nid] for nid in element_nodes])
        surface = area(coords[:,:2])
        B = shape_functions(coords)
        # Element stiffness: K_e = int_{\Omega^e} B^T * C * B d\Omega 
        KE_all[el_id] = 2*surface * B.T @ C @ B
    
    return KE_all

def gen_neighbours(el):
    """
        generate neighbours structure
        in  : elements as triples of node indices
        out : flat neighbour data structure with neigh all the neighbours 
              and neigh_start the start-indices of neigh. 
              ex: neighbours of i = neigh[neigh_start[i]:neigh_start[i+1]]
    """
    t = el
    hedges = np.hstack([
        [t[:,0],t[:,1]],[t[:,1],t[:,0]],
        [t[:,1],t[:,2]],[t[:,2],t[:,1]],
        [t[:,2],t[:,0]],[t[:,0],t[:,2]]]).T
        
    hedges = np.unique(hedges,axis=0).astype(np.int64)
    neigh_start = np.cumsum(np.hstack([[0],np.bincount(hedges[:,0])]))
    neigh = hedges[:,1].copy()

    return neigh, neigh_start

def build_edof_matrix(mesh):
    """Build element DOF matrix from arbitrary mesh (triangular elements)"""
    nel = len(mesh.elements)
    edofMat = np.zeros((nel, 6), dtype=int)  # 6 DOFs per triangle (3 nodes × 2 DOFs)
    
    for el_idx, element_nodes in enumerate(mesh.elements):
        # DOF ordering: [n0_x, n0_y, n1_x, n1_y, n2_x, n2_y]
        dofs = []
        for node_id in element_nodes:
            dofs.extend([2*node_id, 2*node_id + 1])
        edofMat[el_idx, :] = dofs
    
    return edofMat

def boundary_nodes_normals_tangents(nodes, elements):
    nodes = np.asarray(nodes)[:, :2]
    elements = np.asarray(elements, dtype=int)
    centroid = nodes.mean(axis=0)
    edge_count = {}

    for tri in elements:
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted(e))
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    normals = np.zeros_like(nodes, dtype=float)
    tangents = np.zeros_like(nodes, dtype=float)
    boundary_mask = np.zeros(nodes.shape[0], dtype=bool)

    for i, j in boundary_edges:
        pi, pj = nodes[i], nodes[j]
        tangent = pj - pi
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm <= 1e-14:
            continue

        tangent = tangent / tangent_norm
        normal = np.array([tangent[1], -tangent[0]], dtype=float)
        midpoint = 0.5 * (pi + pj)

        if np.dot(normal, midpoint - centroid) < 0.0:
            normal *= -1.0
            tangent *= -1.0

        normals[i] += normal
        normals[j] += normal
        tangents[i] += tangent
        tangents[j] += tangent
        boundary_mask[i] = True
        boundary_mask[j] = True

    boundary_nodes = np.where(boundary_mask)[0]

    nrm = np.linalg.norm(normals[boundary_nodes], axis=1)
    good = nrm > 1e-14
    normals[boundary_nodes[good]] /= nrm[good, None]

    tangents[boundary_nodes] = np.column_stack([-normals[boundary_nodes, 1], normals[boundary_nodes, 0]])

    return boundary_nodes, normals, tangents

def create_mbb_mesh(width, height, mesh_size, F):
    already_initialized = gmsh.isInitialized()
    if not already_initialized:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output
    beam = gmsh.model.occ.addRectangle(0, 0, 0, width, height)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [beam], name="MBB Beam")
    eps = 1e-3  # Tolerance for bounding box

    # Get boundary edges
    left   = gmsh.model.getEntitiesInBoundingBox(-eps,       -eps,        -eps, eps,        height+eps, eps, dim=1)
    right  = gmsh.model.getEntitiesInBoundingBox(width-eps,  -eps,        -eps, width+eps,  height+eps, eps, dim=1)
    bottom = gmsh.model.getEntitiesInBoundingBox(-eps,       -eps,        -eps, width+eps,  eps,        eps, dim=1)
    top    = gmsh.model.getEntitiesInBoundingBox(-eps,       height-eps,  -eps, width+eps,  height+eps, eps, dim=1)

    pleft   = gmsh.model.addPhysicalGroup(1, [tag for _, tag in left],   name="Left")
    pright  = gmsh.model.addPhysicalGroup(1, [tag for _, tag in right],  name="Right")
    pbottom = gmsh.model.addPhysicalGroup(1, [tag for _, tag in bottom], name="Bottom")
    ptop    = gmsh.model.addPhysicalGroup(1, [tag for _, tag in top],    name="Top")

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(2)

    # Get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2, includeBoundary=True)
    node_coords = node_coords.reshape(-1, 3)

    # Create tag to index mapping
    tag2idx = {tag: idx for idx, tag in enumerate(node_tags)}
    tag2idx_func = np.vectorize(tag2idx.get)
    idx2tag = np.array(node_tags)

    # Get elements
    _, _, elements_node = gmsh.model.mesh.getElements(dim=2)
    triangles = elements_node[0].reshape(-1, 3)
    elements = tag2idx_func(triangles)

    # Get left boundary nodes
    left_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pleft)
    left_nodes = tag2idx_func(left_node_tags)

    # Get all border nodes (union of all four edges, deduplicated)
    border_tags = set()
    for pg in [pleft, pright, pbottom, ptop]:
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg)
        border_tags.update(tags)
    border_nodes = tag2idx_func(np.array(list(border_tags)))

    # Find corner nodes (point entities at the four corners)
    corner_coords = [
        (0,     0,      "bottom_left"),
        (width, 0,      "bottom_right"),
        (0,     height, "upper_left"),
        (width, height, "upper_right"),
    ]
    corner_nodes = {}
    corner_ids = []
    for cx, cy, name in corner_coords:
        entities = gmsh.model.getEntitiesInBoundingBox(cx-eps, cy-eps, -eps, cx+eps, cy+eps, eps, dim=0)
        tags = []
        for dim, tag in entities:
            t, _, _ = gmsh.model.mesh.getNodes(dim, tag)
            tags.extend(t)
        corner_nodes[name] = tag2idx_func(np.array(tags))[0]
        corner_ids.append(tag2idx_func(np.array(tags))[0])

    upper_left_node  = corner_nodes["upper_left"]
    bottom_right_node = corner_nodes["bottom_right"]

    if not already_initialized:
        gmsh.finalize()

    # Create mesh with the Mesh class
    mesh = Mesh()
    mesh.elements = elements
    mesh.nodes = node_coords
    mesh.node_tags = idx2tag 

    mesh.areas = np.zeros(len(mesh.elements))
    mesh.B_matrices = np.zeros((len(mesh.elements),3,6))
    for id, element in enumerate(mesh.elements):
        points = np.array([mesh.nodes[i] for i in element])
        mesh.B_matrices[id] = shape_functions(points)
        mesh.areas[id] = area(points)

    mesh.Ke = build_all_element_stiffness_matrices(mesh) # only uses mesh.elements and mesh.nodes
    mesh.edofMat = build_edof_matrix(mesh) # only uses mesh.elements
    mesh.iK = np.repeat(mesh.edofMat, 6, axis=1).ravel()
    mesh.jK = np.tile(mesh.edofMat, (1, 6)).ravel()

    mesh.dirichlet = [(node, 0.0, None) for node in left_nodes]
    mesh.dirichlet.append((bottom_right_node, 0, 0))
    mesh.neumann = [(upper_left_node, 0.0, F)]

    stiffness, M, rad = helmholtz_filter(mesh)
    mesh.stiff_matrix = stiffness
    mesh.mass_matrix = M
    mesh.r = rad

    # Border and corner node collections
    mesh.borders = np.array(border_nodes)
    mesh.corners = np.array(corner_ids)
    mesh.neigh, mesh.neighstart = gen_neighbours(mesh.elements)
    _, mesh.boundary_normals, mesh.boundary_tangents = boundary_nodes_normals_tangents(mesh.nodes, mesh.elements)

    return mesh

def create_connecting_rod_mesh(length=3.0, r_big_outer=0.45, r_big_inner=0.2, r_small_outer=0.28, r_small_inner=0.13, shank_width=0.32, mesh_size=0.02, F=-4000*9.81, fix_big_end=True, load_small_end=True):
    already_initialized = gmsh.isInitialized()
    if not already_initialized: gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.add("connecting_rod")

    big_center = np.array([0.0, 0.0])
    small_center = np.array([length, 0.0])
    eps = 1e-6

    big_outer = gmsh.model.occ.addDisk(big_center[0], big_center[1], 0, r_big_outer, r_big_outer)
    small_outer = gmsh.model.occ.addDisk(small_center[0], small_center[1], 0, r_small_outer, r_small_outer)
    shank = gmsh.model.occ.addRectangle(big_center[0], -shank_width/2, 0, length, shank_width)

    outer, _ = gmsh.model.occ.fuse([(2, big_outer)], [(2, small_outer), (2, shank)])
    outer_surfaces = outer

    big_hole = gmsh.model.occ.addDisk(big_center[0], big_center[1], 0, r_big_inner, r_big_inner)
    small_hole = gmsh.model.occ.addDisk(small_center[0], small_center[1], 0, r_small_inner, r_small_inner)

    rod, _ = gmsh.model.occ.cut(outer_surfaces, [(2, big_hole), (2, small_hole)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    surface_tags = [tag for dim, tag in rod if dim == 2]
    gmsh.model.addPhysicalGroup(2, surface_tags, name="Connecting rod")

    curves = gmsh.model.getEntities(1)
    big_inner_curves, small_inner_curves, outer_curves = [], [], []

    for dim, tag in curves:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        d_big = np.linalg.norm(np.array([cx, cy]) - big_center)
        d_small = np.linalg.norm(np.array([cx, cy]) - small_center)

        if d_big < 1.25 * r_big_inner:
            big_inner_curves.append(tag)
        elif d_small < 1.25 * r_small_inner:
            small_inner_curves.append(tag)
        else:
            outer_curves.append(tag)

    pbig_inner = gmsh.model.addPhysicalGroup(1, big_inner_curves, name="Big inner hole")
    psmall_inner = gmsh.model.addPhysicalGroup(1, small_inner_curves, name="Small inner hole")
    pouter = gmsh.model.addPhysicalGroup(1, outer_curves, name="Outer boundary")

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2, includeBoundary=True)
    node_coords = node_coords.reshape(-1, 3)

    tag2idx = {tag: idx for idx, tag in enumerate(node_tags)}
    tag2idx_func = np.vectorize(tag2idx.get)
    idx2tag = np.array(node_tags)

    _, _, elements_node = gmsh.model.mesh.getElements(dim=2)
    triangles = elements_node[0].reshape(-1, 3)
    elements = tag2idx_func(triangles)

    big_inner_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pbig_inner)
    small_inner_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, psmall_inner)
    outer_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pouter)

    big_inner_nodes = tag2idx_func(big_inner_tags)
    small_inner_nodes = tag2idx_func(small_inner_tags)
    outer_nodes = tag2idx_func(outer_tags)
    border_nodes = np.unique(np.concatenate([big_inner_nodes, small_inner_nodes, outer_nodes]))

    mesh = Mesh()
    mesh.elements = elements
    mesh.nodes = node_coords
    mesh.node_tags = idx2tag

    mesh.areas = np.zeros(len(mesh.elements))
    mesh.B_matrices = np.zeros((len(mesh.elements), 3, 6))

    for el_id, element in enumerate(mesh.elements):
        points = np.array([mesh.nodes[i] for i in element])
        mesh.B_matrices[el_id] = shape_functions(points)
        mesh.areas[el_id] = area(points)

    mesh.Ke = build_all_element_stiffness_matrices(mesh)
    mesh.edofMat = build_edof_matrix(mesh)
    mesh.iK = np.repeat(mesh.edofMat, 6, axis=1).ravel()
    mesh.jK = np.tile(mesh.edofMat, (1, 6)).ravel()

    mesh.dirichlet = []
    mesh.neumann = []

    if fix_big_end:
        mesh.dirichlet = [(node, 0.0, 0.0) for node in big_inner_nodes]

    if load_small_end:
        force_per_node = F / max(len(small_inner_nodes), 1)
        mesh.neumann = [(node, 0.0, force_per_node) for node in small_inner_nodes]

    stiffness, M, rad = helmholtz_filter(mesh)
    mesh.stiff_matrix = stiffness
    mesh.mass_matrix = M
    mesh.r = rad

    mesh.borders = border_nodes
    mesh.corners = np.array([], dtype=int)
    mesh.neigh, mesh.neighstart = gen_neighbours(mesh.elements)

    if not already_initialized: gmsh.finalize()

    return mesh

def create_crane_mesh(width=30, height=20, mast_width=2.5, beam_height=1.5, triangle_height=4.0, mesh_size=0.15, F=-4000*9.81):
    already_initialized = gmsh.isInitialized()
    if not already_initialized: gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("crane")

    eps = 1e-6
    y_beam_min = height - beam_height / 2
    y_beam_max = height + beam_height / 2

    mast = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, mast_width, height)
    beam = gmsh.model.occ.addRectangle(-width/5.0, y_beam_min, 0.0, width, beam_height)

    p1 = gmsh.model.occ.addPoint(-width/5.0, y_beam_max, 0.0)
    p2 = gmsh.model.occ.addPoint(width*0.8, y_beam_max, 0.0)
    p3 = gmsh.model.occ.addPoint(mast_width, y_beam_max + triangle_height, 0.0)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p1)

    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3])
    triangle = gmsh.model.occ.addPlaneSurface([loop])

    domain, _ = gmsh.model.occ.fuse([(2, mast)], [(2, beam), (2, triangle)])
    gmsh.model.occ.synchronize()

    surface_tags = [tag for dim, tag in domain if dim == 2]
    gmsh.model.addPhysicalGroup(2, surface_tags, name="Crane")

    bottom = gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps, mast_width + eps, eps, eps, dim=1)
    right_tip = gmsh.model.getEntitiesInBoundingBox(width - eps, y_beam_min - eps, -eps, width + eps, y_beam_max + eps, eps, dim=0)

    pbottom = gmsh.model.addPhysicalGroup(1, [tag for _, tag in bottom], name="Bottom support")

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2, includeBoundary=True)
    node_coords = node_coords.reshape(-1, 3)

    tag2idx = {tag: idx for idx, tag in enumerate(node_tags)}
    tag2idx_func = np.vectorize(tag2idx.get)
    idx2tag = np.array(node_tags)

    _, _, elements_node = gmsh.model.mesh.getElements(dim=2)
    triangles = elements_node[0].reshape(-1, 3)
    elements = tag2idx_func(triangles)

    bottom_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pbottom)
    bottom_nodes = tag2idx_func(bottom_node_tags)

    all_curves = gmsh.model.getEntities(1)
    border_tags = set()
    for _, curve_tag in all_curves:
        tags, _, _ = gmsh.model.mesh.getNodes(1, curve_tag)
        border_tags.update(tags)
    border_nodes = tag2idx_func(np.array(list(border_tags)))

    corner_coordinates = np.array([[0.0, 0.0], [mast_width, 0.0], [0.0, y_beam_max], [width, y_beam_max], [mast_width, y_beam_max + triangle_height], [width, y_beam_min]])
    corner_ids = []
    for coord in corner_coordinates:
        distances = np.linalg.norm(node_coords[:, :2] - coord[None, :], axis=1)
        corner_ids.append(np.argmin(distances))

    load_point = np.array([width, y_beam_min])
    load_node = np.argmin(np.linalg.norm(node_coords[:, :2] - load_point[None, :], axis=1))

    if not already_initialized: gmsh.finalize()

    mesh = Mesh()
    mesh.elements = elements
    mesh.nodes = node_coords
    mesh.node_tags = idx2tag

    mesh.areas = np.zeros(len(mesh.elements))
    mesh.B_matrices = np.zeros((len(mesh.elements), 3, 6))
    for el_id, element in enumerate(mesh.elements):
        points = np.array([mesh.nodes[i] for i in element])
        mesh.B_matrices[el_id] = shape_functions(points)
        mesh.areas[el_id] = area(points)

    mesh.Ke = build_all_element_stiffness_matrices(mesh)
    mesh.edofMat = build_edof_matrix(mesh)
    mesh.iK = np.repeat(mesh.edofMat, 6, axis=1).ravel()
    mesh.jK = np.tile(mesh.edofMat, (1, 6)).ravel()

    mesh.dirichlet = [(node, 0.0, 0.0) for node in bottom_nodes]
    mesh.neumann = [(load_node, 0.0, F)]

    stiffness, M, rad = helmholtz_filter(mesh)
    mesh.stiff_matrix = stiffness
    mesh.mass_matrix = M
    mesh.r = rad

    mesh.borders = np.array(border_nodes)
    mesh.corners = np.array(corner_ids)
    mesh.neigh, mesh.neighstart = gen_neighbours(mesh.elements)

    return mesh

def create_bridge_mesh(width=20.0, height=1.0, mesh_size=0.1, total_load=-4000*9.81):
    already_initialized = gmsh.isInitialized()
    if not already_initialized: gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("bridge")

    eps = 1e-6
    bridge = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, width, height)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [bridge], name="Bridge")

    left = gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps, eps, height + eps, eps, dim=1)
    right = gmsh.model.getEntitiesInBoundingBox(width - eps, -eps, -eps, width + eps, height + eps, eps, dim=1)
    bottom = gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps, width + eps, eps, eps, dim=1)
    top = gmsh.model.getEntitiesInBoundingBox(-eps, height - eps, -eps, width + eps, height + eps, eps, dim=1)

    ptop = gmsh.model.addPhysicalGroup(1, [tag for _, tag in top], name="Top")
    pbottom = gmsh.model.addPhysicalGroup(1, [tag for _, tag in bottom], name="Bottom")
    gmsh.model.addPhysicalGroup(1, [tag for _, tag in left], name="Left")
    gmsh.model.addPhysicalGroup(1, [tag for _, tag in right], name="Right")

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2, includeBoundary=True)
    node_coords = node_coords.reshape(-1, 3)

    tag2idx = {tag: idx for idx, tag in enumerate(node_tags)}
    tag2idx_func = np.vectorize(tag2idx.get)
    idx2tag = np.array(node_tags)

    _, _, elements_node = gmsh.model.mesh.getElements(dim=2)
    triangles = elements_node[0].reshape(-1, 3)
    elements = tag2idx_func(triangles)

    top_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, ptop)
    top_nodes = tag2idx_func(top_node_tags)

    border_tags = set()
    for pg in [ptop, pbottom]:
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg)
        border_tags.update(tags)
    for entity_group in [left, right]:
        for _, tag in entity_group:
            tags, _, _ = gmsh.model.mesh.getNodes(1, tag)
            border_tags.update(tags)
    border_nodes = tag2idx_func(np.array(list(border_tags)))

    bottom_left_node = np.argmin(np.linalg.norm(node_coords[:, :2] - np.array([0.0, 0.0]), axis=1))
    bottom_right_node = np.argmin(np.linalg.norm(node_coords[:, :2] - np.array([width, 0.0]), axis=1))
    upper_left_node = np.argmin(np.linalg.norm(node_coords[:, :2] - np.array([0.0, height]), axis=1))
    upper_right_node = np.argmin(np.linalg.norm(node_coords[:, :2] - np.array([width, height]), axis=1))

    if not already_initialized: gmsh.finalize()

    mesh = Mesh()
    mesh.elements = elements
    mesh.nodes = node_coords
    mesh.node_tags = idx2tag

    mesh.areas = np.zeros(len(mesh.elements))
    mesh.B_matrices = np.zeros((len(mesh.elements), 3, 6))
    for el_id, element in enumerate(mesh.elements):
        points = np.array([mesh.nodes[i] for i in element])
        mesh.B_matrices[el_id] = shape_functions(points)
        mesh.areas[el_id] = area(points)

    mesh.Ke = build_all_element_stiffness_matrices(mesh)
    mesh.edofMat = build_edof_matrix(mesh)
    mesh.iK = np.repeat(mesh.edofMat, 6, axis=1).ravel()
    mesh.jK = np.tile(mesh.edofMat, (1, 6)).ravel()

    mesh.dirichlet = [(bottom_left_node, 0.0, 0.0), (bottom_right_node, 0.0, 0.0)]
    mesh.dirichlet += [(node, 0.0, None) for node in top_nodes]

    force_per_node = total_load / len(top_nodes)
    mesh.neumann = [(node, 0.0, force_per_node) for node in top_nodes]

    stiffness, M, rad = helmholtz_filter(mesh)
    mesh.stiff_matrix = stiffness
    mesh.mass_matrix = M
    mesh.r = rad

    mesh.borders = np.array(border_nodes)
    mesh.corners = np.array([bottom_left_node, bottom_right_node, upper_left_node, upper_right_node])
    mesh.neigh, mesh.neighstart = gen_neighbours(mesh.elements)

    return mesh

def create_wishbone_mesh(scale=1.0, length=3.0, mount_sep=1.0, r_mount_outer=0.28, r_mount_inner=0.12, r_ball_outer=0.35, r_ball_inner=0.15, arm_width=0.28, mesh_size=0.04, F=-4000*9.81):
    
    length *= scale
    mount_sep *= scale
    r_mount_outer *= scale
    r_mount_inner *= scale
    r_ball_outer *= scale
    r_ball_inner *= scale
    arm_width *= scale

    already_initialized = gmsh.isInitialized()
    if not already_initialized: gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("wishbone")

    left_top = np.array([0.0, mount_sep / 2])
    left_bottom = np.array([0.0, -mount_sep / 2])
    right = np.array([length, 0.0])

    top_outer = gmsh.model.occ.addDisk(left_top[0], left_top[1], 0.0, r_mount_outer, r_mount_outer)
    bottom_outer = gmsh.model.occ.addDisk(left_bottom[0], left_bottom[1], 0.0, r_mount_outer, r_mount_outer)
    ball_outer = gmsh.model.occ.addDisk(right[0], right[1], 0.0, r_ball_outer, r_ball_outer)

    top_arm = gmsh.model.occ.addRectangle(0.0, left_top[1] - arm_width / 2, 0.0, length, arm_width)
    gmsh.model.occ.rotate([(2, top_arm)], 0.0, left_top[1], 0.0, 0.0, 0.0, 1.0, np.arctan2(right[1] - left_top[1], right[0] - left_top[0]))

    bottom_arm = gmsh.model.occ.addRectangle(0.0, left_bottom[1] - arm_width / 2, 0.0, length, arm_width)
    gmsh.model.occ.rotate([(2, bottom_arm)], 0.0, left_bottom[1], 0.0, 0.0, 0.0, 1.0, np.arctan2(right[1] - left_bottom[1], right[0] - left_bottom[0]))

    cross_member = gmsh.model.occ.addRectangle(-r_mount_outer, -arm_width / 2, 0.0, 2 * r_mount_outer, arm_width)
    gmsh.model.occ.rotate([(2, cross_member)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.pi / 2)

    domain, _ = gmsh.model.occ.fuse([(2, top_outer)], [(2, bottom_outer), (2, ball_outer), (2, top_arm), (2, bottom_arm), (2, cross_member)])
    gmsh.model.occ.synchronize()

    top_hole = gmsh.model.occ.addDisk(left_top[0], left_top[1], 0.0, r_mount_inner, r_mount_inner)
    bottom_hole = gmsh.model.occ.addDisk(left_bottom[0], left_bottom[1], 0.0, r_mount_inner, r_mount_inner)
    ball_hole = gmsh.model.occ.addDisk(right[0], right[1], 0.0, r_ball_inner, r_ball_inner)

    wishbone, _ = gmsh.model.occ.cut(domain, [(2, top_hole), (2, bottom_hole), (2, ball_hole)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    surface_tags = [tag for dim, tag in wishbone if dim == 2]
    gmsh.model.addPhysicalGroup(2, surface_tags, name="Wishbone")

    curves = gmsh.model.getEntities(1)
    top_inner_curves, bottom_inner_curves, ball_inner_curves, outer_curves = [], [], [], []

    for dim, tag in curves:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        p = np.array([cx, cy])
        if np.linalg.norm(p - left_top) < 1.4 * r_mount_inner:
            top_inner_curves.append(tag)
        elif np.linalg.norm(p - left_bottom) < 1.4 * r_mount_inner:
            bottom_inner_curves.append(tag)
        elif np.linalg.norm(p - right) < 1.4 * r_ball_inner:
            ball_inner_curves.append(tag)
        else:
            outer_curves.append(tag)

    ptop_inner = gmsh.model.addPhysicalGroup(1, top_inner_curves, name="Top mount hole")
    pbottom_inner = gmsh.model.addPhysicalGroup(1, bottom_inner_curves, name="Bottom mount hole")
    pball_inner = gmsh.model.addPhysicalGroup(1, ball_inner_curves, name="Ball joint hole")
    pouter = gmsh.model.addPhysicalGroup(1, outer_curves, name="Outer boundary")

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2, includeBoundary=True)
    node_coords = node_coords.reshape(-1, 3)

    tag2idx = {tag: idx for idx, tag in enumerate(node_tags)}
    tag2idx_func = np.vectorize(tag2idx.get)
    idx2tag = np.array(node_tags)

    _, _, elements_node = gmsh.model.mesh.getElements(dim=2)
    triangles = elements_node[0].reshape(-1, 3)
    elements = tag2idx_func(triangles)

    top_inner_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, ptop_inner)
    bottom_inner_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pbottom_inner)
    ball_inner_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pball_inner)
    outer_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pouter)

    top_inner_nodes = tag2idx_func(top_inner_tags)
    bottom_inner_nodes = tag2idx_func(bottom_inner_tags)
    ball_inner_nodes = tag2idx_func(ball_inner_tags)
    outer_nodes = tag2idx_func(outer_tags)
    border_nodes = np.unique(np.concatenate([top_inner_nodes, bottom_inner_nodes, ball_inner_nodes, outer_nodes]))

    mesh = Mesh()
    mesh.elements = elements
    mesh.nodes = node_coords
    mesh.node_tags = idx2tag

    mesh.areas = np.zeros(len(mesh.elements))
    mesh.B_matrices = np.zeros((len(mesh.elements), 3, 6))
    for el_id, element in enumerate(mesh.elements):
        points = np.array([mesh.nodes[i] for i in element])
        mesh.B_matrices[el_id] = shape_functions(points)
        mesh.areas[el_id] = area(points)

    mesh.Ke = build_all_element_stiffness_matrices(mesh)
    mesh.edofMat = build_edof_matrix(mesh)
    mesh.iK = np.repeat(mesh.edofMat, 6, axis=1).ravel()
    mesh.jK = np.tile(mesh.edofMat, (1, 6)).ravel()

    mesh.dirichlet = [(node, 0.0, 0.0) for node in np.unique(np.concatenate([top_inner_nodes, bottom_inner_nodes]))]

    force_per_node = F / max(len(ball_inner_nodes), 1)
    mesh.neumann = [(node, 0.0, force_per_node) for node in ball_inner_nodes]

    stiffness, M, rad = helmholtz_filter(mesh)
    mesh.stiff_matrix = stiffness
    mesh.mass_matrix = M
    mesh.r = rad

    mesh.borders = border_nodes
    mesh.corners = np.array([], dtype=int)
    mesh.neigh, mesh.neighstart = gen_neighbours(mesh.elements)

    if not already_initialized: gmsh.finalize()
    return mesh