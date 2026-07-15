import gmsh
import math 
import numpy as np
import ui_module as ui
import mesh_struct as ms
from itertools import combinations_with_replacement
from scipy.sparse.linalg import spsolve, factorized
from scipy.sparse import coo_matrix

def barycentric_integral(density_element, p):
    """
    Compute the exact element average of rho**p over linear tetrahedra.

    Args: 
    - density_element : ndarray, shape (nel, 4) --> Nodal density values of each tetrahedron.
    - p : int --> SIMP exponent.

    Returns: 
    - average : ndarray, shape (nel,) --> Exact values of (1 / V_e) * integral_e rho**p dV.
    """
    if p < 0: raise ValueError("p must be non-negative.")
    if p == 0: return np.ones(len(density_element))

    average = np.zeros(len(density_element))

    for indices in combinations_with_replacement(range(4), p):
        term = np.ones(len(density_element))
        for index in indices:
            term *= density_element[:, index]
        average += term

    return average / float(math.comb(p + 3, 3))

def barycentric_integral_derivative(density_element, p):
    """
    Compute the derivative of the exact tetrahedral average of rho**p with respect to the four nodal densities.

    Returns:
    - derivative : ndarray, shape (nel, 4)
    """
    density_element = np.asarray(density_element, dtype=float)
    derivative = np.zeros_like(density_element)

    if p == 0: return derivative
    
    for indices in combinations_with_replacement(range(4), p):
        multiplicities = np.bincount(indices, minlength=4)

        for node in range(4):
            if multiplicities[node] == 0: continue

            term = multiplicities[node] * np.ones(len(density_element))
            for variable in range(4):
                exponent = multiplicities[variable] - (1 if variable == node else 0)
                if exponent > 0:
                    term *= density_element[:, variable]**exponent

            derivative[:, node] += term

    return derivative / float(math.comb(p + 3, 3))

def fem_solver(mesh, density, p):
    """
    Assemble and solve the 3D elasticity problem on linear tetrahedra.
    Dirichlet conditions enforced by direct elimination.
    """
    num_nodes = len(mesh.nodes)
    num_dofs = 3 * num_nodes

    density = np.asarray(density, dtype=float)
    density_el = density[mesh.elements]

    rho_p_average = barycentric_integral(density_el, p)
    element_Young = ms.E_void + (ms.E_matter - ms.E_void) * rho_p_average

    sK = (mesh.Ke.reshape(-1, 144) * element_Young[:, None]).ravel()
    K = coo_matrix((sK, (mesh.iK, mesh.jK)), shape=(num_dofs, num_dofs)).tocsr()

    mass_pattern = np.array([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 2.0]
    ]) / 20.0

    local_gravity = density_el @ mass_pattern
    local_gravity *= -ms.rho_matter * ms.gravity * mesh.volumes[:, None]

    rhs_z = np.zeros(num_nodes)
    np.add.at(rhs_z, mesh.elements, local_gravity)

    rhs = np.zeros(num_dofs)
    rhs[2::3] = rhs_z

    for condition in mesh.neumann:
        if len(condition) == 4:
            node_id, fx, fy, fz = condition
            rhs[3 * node_id] += 0.0 if fx is None else fx
            rhs[3 * node_id + 1] += 0.0 if fy is None else fy
            rhs[3 * node_id + 2] += 0.0 if fz is None else fz

        elif len(condition) == 5:
            node_id, fn, ft1, ft2, frame = condition

            if frame != "ntt":
                raise ValueError(f"Unknown 3D force frame {frame!r}.")

            normal = np.asarray(mesh.boundary_normals[node_id], dtype=float)
            tangent_1 = np.asarray(mesh.boundary_tangent_1[node_id], dtype=float)
            tangent_2 = np.asarray(mesh.boundary_tangent_2[node_id], dtype=float)

            if np.linalg.norm(normal) <= 1e-14:
                raise ValueError(f"No valid boundary normal at node {node_id}.")

            normal /= np.linalg.norm(normal)
            tangent_1 /= np.linalg.norm(tangent_1)
            tangent_2 /= np.linalg.norm(tangent_2)

            force = np.zeros(3)
            if fn is not None:
                force += fn * normal
            if ft1 is not None:
                force += ft1 * tangent_1
            if ft2 is not None:
                force += ft2 * tangent_2

            rhs[3 * node_id:3 * node_id + 3] += force

        else:
            raise ValueError(f"Invalid Neumann condition: {condition}")

    # Direct elimination of Dirichlet dofs.
    fixed_dofs = []
    fixed_values = []

    for node_id, ux, uy, uz in mesh.dirichlet:
        for component, value in enumerate((ux, uy, uz)):
            if value is None:
                continue
            fixed_dofs.append(3 * node_id + component)
            fixed_values.append(value)

    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    fixed_values = np.asarray(fixed_values, dtype=float)

    u = np.zeros(num_dofs)

    if len(fixed_dofs) == 0:
        u = spsolve(K, rhs)
    else:
        all_dofs = np.arange(num_dofs)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=False)

        u[fixed_dofs] = fixed_values

        K_ff = K[free_dofs][:, free_dofs]
        K_fc = K[free_dofs][:, fixed_dofs]

        rhs_free = rhs[free_dofs] - K_fc @ fixed_values

        u[free_dofs] = spsolve(K_ff, rhs_free)

    return u, K, rhs


def dcompliance_drho(mesh, density, u, p):
    num_nodes = len(mesh.nodes)
    density_el = density[mesh.elements]
    ue = u[mesh.edofMat]

    unit_energy = np.einsum("ei,eij,ej->e", ue, mesh.Ke, ue)
    daverage_drho = barycentric_integral_derivative(density_el, p)

    local_stiffness_gradient = -(ms.E_matter - ms.E_void) * unit_energy[:, None] * daverage_drho

    gradient = np.zeros(num_nodes)
    np.add.at(gradient, mesh.elements, local_stiffness_gradient)

    mass_pattern = np.array([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 2.0]
    ]) / 20.0

    uz_el = u[3 * mesh.elements + 2]
    local_load_gradient = -ms.rho_matter * ms.gravity * mesh.volumes[:, None] * (uz_el @ mass_pattern)
    np.add.at(gradient, mesh.elements, 2.0 * local_load_gradient)

    return gradient

def density_approach(mesh, volfrac, radius=None, p=3, max_iter=1000, init_design=None, plot=True, outfile=None, log=False):
    """
    Nodal-density topology optimization on a tetrahedral mesh using
    a Helmholtz PDE filter.
    """
    threshold = 2e-3

    if plot:
        print("3D topology optimization with Helmholtz PDE filter")
        print(f"Elements: {len(mesh.elements)}, Nodes: {len(mesh.nodes)}")
        print(f"Volume fraction: {volfrac}, p: {p}")

    nNodes = len(mesh.nodes)

    design = volfrac * np.ones(nNodes) if init_design is None else init_design.copy()
    density = design.copy()
    total_volume = np.sum(mesh.volumes)

    M_filter = mesh.mass_matrix
    radius = mesh.r if radius is None else radius
    H_operator = radius**2 * mesh.stiff_matrix + M_filter
    H_solver = factorized(H_operator.tocsc())

    view = ui.DensityView(mesh.node_tags, mesh.elements) if plot else None

    compliances = []
    volume_fractions = []
    densities = []
    variations = []

    # dV/drho for linear tetrahedral interpolation.
    dV_drho = np.zeros(nNodes)
    np.add.at(dV_drho, mesh.elements.ravel(), np.repeat(mesh.volumes / 4.0, 4))

    # density = H^-1 M design
    # dV/ddesign = M.T H^-T dV/drho
    volume_adjoint = H_solver(dV_drho)
    dV_ddesign = np.asarray(M_filter.T @ volume_adjoint).ravel()

    iteration = 0
    change = 1.0
    compliance = np.nan
    u = np.zeros(3*nNodes)
    rhs = np.zeros(3*nNodes)

    while change > threshold and iteration < max_iter:
        density = np.clip(H_solver(M_filter @ design), 0.0, 1.0)

        u, _, rhs = fem_solver(mesh, density, p)
        compliance = float(u @ rhs)

        dc_drho = dcompliance_drho(mesh, density, u, p)

        compliance_adjoint = H_solver(dc_drho)
        dc_ddesign = np.asarray(M_filter.T @ compliance_adjoint).ravel()

        old_design = design.copy()
        design = oc(design, dc_ddesign, dV_ddesign, volfrac * total_volume)

        change = np.linalg.norm(design - old_design) / max(np.linalg.norm(old_design), 1e-14)
        physical_volume = np.dot(mesh.volumes, np.mean(density[mesh.elements], axis=1))
        volume_fraction = physical_volume / total_volume

        compliances.append(compliance)
        volume_fractions.append(volume_fraction)
        densities.append(density.copy())
        variations.append(change)

        iteration += 1

        if plot:
            print(f"  iteration {iteration:4d}  |  compliance = {compliance:.4e}  |  volume = {volume_fraction:.4f}  |  design variation = {change:.4f}")
            view.update(mesh.nodes, density, iteration=iteration, compliance=compliance)

    if plot:
        if iteration == 0:
            density = np.clip(H_solver(M_filter @ design), 0.0, 1.0)
            u, _, rhs = fem_solver(mesh, density, p)
            compliance = float(u @ rhs)

        view.update(mesh.nodes, density, iteration=iteration, compliance=compliance)

        if gmsh.fltk.isAvailable(): gmsh.fltk.run()

        duration = 8.0 # duration of animation
        fps = int((iteration+1)/duration)
        # if outfile is not None: view.save_mp4(outfile, fps)
        # view.show_interactive(iso_value=0.5, wait=True)

    log_data = [compliances, volume_fractions, densities, variations] if log else None

    return density, iteration, u, rhs, log_data

def oc(x, dc, dv, vol_target):
    l1 = 0.0
    l2 = 1e9
    move = 0.2

    while (l2 - l1) / (l2 + 1e-12) > 1e-3:
        lmid = 0.5 * (l1 + l2)
        ratio = np.maximum(-dc / np.maximum(dv * lmid, 1e-30), 1e-12)
        xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(ratio)))))

        if np.dot(dv, xnew) > vol_target: l1 = lmid
        else: l2 = lmid

    return xnew

if __name__ == "__main__":
    p = 3
    density = np.array([[0.2, 0.4, 0.7, 1.0], [0.1, 0.3, 0.5, 0.8]])
    
    print("\nTests for barycentric integral")
    print(barycentric_integral(density, p))
    print("\n")
    
    print("Tests for barycentric integral derivative")
    print(barycentric_integral_derivative(density, p))
    print("\n")
    if not gmsh.isInitialized():
        gmsh.initialize()

    h = 0.1
    volfrac = 0.10
    outfile = "outputs/to_3d.mp4"
    mesh = ms.create_mbb_mesh(3.0, 1.0, h, -4e6*9.81)
    density = np.ones(len(mesh.nodes))
    u, K, rhs = fem_solver(mesh, density, p)
    
    print(f"\nNumber of nodes: {len(mesh.nodes)}")
    print(f"Number of DOFs: {3*len(mesh.nodes)}")
    print(f"Number of DOFs per node: {3}")
    print(f"K nnz: {K.nnz}\n")
    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    print(f"max |u_x| = {np.linalg.norm(ux, np.inf):.3e}")
    print(f"max |u_y| = {np.linalg.norm(uy, np.inf):.3e}")
    print(f"max |u_z| = {np.linalg.norm(uz, np.inf):.3e}")
    
    print(f"Number of Dirichlet BC nodes: {len(mesh.dirichlet)}")
    print(f"Number of Neumann BC nodes: {len(mesh.neumann)}\n")
    
    print(f"sum rhs x: {np.sum(rhs[0::3]):.3e}")
    print(f"sum rhs y: {np.sum(rhs[1::3]):.3e}")
    print(f"sum rhs z: {np.sum(rhs[2::3]):.3e}")
    print(f"norm rhs: {np.linalg.norm(rhs):.3e}")
    print("\n\n")
    
    expected_gravity = 0.0

    for element, nodes in enumerate(mesh.elements):
        mean_density = np.mean(density[nodes])
        expected_gravity -= ms.rho_matter * ms.gravity * mesh.volumes[element] * mean_density

    print(f"expected gravity   = {expected_gravity:.3e}")
    print(f"actual gravity     = {np.sum(rhs[2::3]) - sum(c[3] for c in mesh.neumann):.3e}")
    
    print(f"\nmax |K_ij| = {np.max(np.abs(K.data)):.3e}")
    print(f"max |K_ii| = {np.max(np.abs(K.diagonal())):.3e}")

    density, iterations, u, rhs, logs = density_approach(mesh, volfrac, p=3, outfile=outfile, log=True, max_iter=1000)

    if gmsh.isInitialized():
        gmsh.finalize()