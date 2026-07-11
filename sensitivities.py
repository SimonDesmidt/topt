import math 
import numpy as np
import mesh_struct as ms
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from itertools import combinations_with_replacement
from scipy.sparse import coo_matrix

## FEM solver ###################################################################################################################

def barycentric_integral(density_element, p):
    """
    Compute the exact element average of rho**p over linear tetrahedra.

    Parameters
    ----------
    density_element : ndarray, shape (nel, 4)
        Nodal density values of each tetrahedron.
    p : int
        SIMP exponent.

    Returns
    -------
    average : ndarray, shape (nel,)
        Exact values of (1 / V_e) * integral_e rho**p dV.
    """
    density_element = np.asarray(density_element, dtype=float)
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
    Exact gradient of (1/V_e) * integral_e rho**p dV
    w.r.t. each of the 4 nodal densities.

    Returns
    -------
    grad : ndarray, shape (n_elements, 4)
    """
    density_element = np.asarray(density_element, dtype=float)
    p = int(p)
    nel = len(density_element)
    grad = np.zeros((nel, 4))
    if p == 0: return grad

    for combo in combinations_with_replacement(range(4), p):
        for pos in range(p):
            idx = combo[pos]
            reduced = combo[:pos] + combo[pos + 1:]
            term = np.ones(nel)
            for k in reduced:
                term *= density_element[:, k]
            grad[:, idx] += term

    return grad / float(math.comb(p + 3, 3))

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

        u[free_dofs] = spsolve(K_ff.tocsr(), rhs_free)

    return u

## Sensitivity computation ######################################################################################################

def get_drhs_drho(mesh):
    """
    Compute the derivative of the 3D FEM right-hand side with respect
    to the nodal density.

    The body force is assumed to be:
        f_y = -rho_matter * gravity * integral(N_i * rho dV)

    Returns
    -------
    df_drho : scipy.sparse.csr_matrix, shape (3*nNodes, nNodes)
        Derivative of the global force vector with respect to nodal density.
    """
    numNodes = len(mesh.nodes)
    df_drho = sp.lil_matrix((3 * numNodes, numNodes))

    # Consistent mass matrix for a linear tetrahedron:
    # M_e = V/20 * [[2,1,1,1], ...]
    parent_mass = np.array([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 2.0]
    ]) / 20.0

    for element_id, element in enumerate(mesh.elements):
        jac = mesh.volumes[element_id] 
        for i, node_i in enumerate(element):
            for j, node_j in enumerate(element):
                df_drho[3 * node_i + 2, node_j] += parent_mass[i, j] * jac 
    df_drho *= -ms.rho_matter * ms.gravity 

    return df_drho.tocsr()

def get_dKu_drho(mesh, u, density, p):
    """Compute d(Ku)/d(rho) for linear tetrahedral elements."""
    numNodes = len(mesh.nodes)
    dK_drho_u = sp.lil_matrix((3 * numNodes, numNodes))

    density = np.asarray(density, dtype=float)
    density_el = density[mesh.elements]
    grad_rho_p_avg = barycentric_integral_derivative(density_el, p)  # (nel, 4)

    for element_id, element in enumerate(mesh.elements):
        B = mesh.B_matrices[element_id]
        volume = mesh.volumes[element_id]

        ue = np.zeros(12)
        for i, ni in enumerate(element):
            ue[3*i]   = u[3*ni]
            ue[3*i+1] = u[3*ni+1]
            ue[3*i+2] = u[3*ni+2]

        BTCBu = volume * (B.T @ ms.C @ B @ ue)

        dE_drho_local = (ms.E_matter - ms.E_void) * grad_rho_p_avg[element_id]
        dKe_drhoe_u = np.outer(BTCBu, dE_drho_local)

        for i, ni in enumerate(element):
            for j, nj in enumerate(element):
                dK_drho_u[3*ni, nj]   += dKe_drhoe_u[3*i, j]
                dK_drho_u[3*ni+1, nj] += dKe_drhoe_u[3*i+1, j]
                dK_drho_u[3*ni+2, nj] += dKe_drhoe_u[3*i+2, j]

    return dK_drho_u.tocsr()

def sensitivity_compliance(mesh, control, p):
    """ Compute the sensitivity of the compliance with respect to the control variable. """
    r = mesh.r
    M = mesh.mass_matrix 
    H = r*r * mesh.stiff_matrix + M
    density = spsolve(H, M@control**p)
    u = fem_solver(mesh, density, p)
    lmbda = -2*u # spsolve(K.T, -(K+K.T)@u) # Adjoint parameter 

    drho_dk = spsolve(H, p*M@np.diag(control**(p-1)))
    df_drho = get_drhs_drho(mesh) # Derivative of the rhs term 
    dK_drho_u = get_dKu_drho(mesh, u, density, p) # Derivative of the stiffness matrix K multiplied by the displacement vector u : dK_drho @ u
    return ((u+lmbda).T @ dK_drho_u - lmbda.T @ df_drho) @ drho_dk 

## Volume constraint and its gradient ###########################################################################################

def volume_constraint(mesh, control, alpha, p=3):
    """
    Compute the volume constraint: V(rho) - alpha * V_total <= 0
    with Helmholtz filter: H rho = M control**p
    """
    control = np.asarray(control, dtype=float)

    M = mesh.mass_matrix
    H = mesh.r**2 * mesh.stiff_matrix + M
    density = spsolve(H, M @ control**p)

    material_volume = 0.0
    total_volume = np.sum(mesh.volumes)

    for element_id, element in enumerate(mesh.elements):
        material_volume += mesh.volumes[element_id] * np.mean(density[element])

    return material_volume - alpha * total_volume

def sensitivity_volume(mesh, control, p):
    """
    Compute the gradient of the volume constraint with respect to control.
    """
    control = np.asarray(control, dtype=float)
    num_nodes = len(mesh.nodes)

    dV_drho = np.zeros(num_nodes)

    for element_id, element in enumerate(mesh.elements):
        dV_drho[element] += mesh.volumes[element_id] / 4.0

    M = mesh.mass_matrix
    H = mesh.r**2 * mesh.stiff_matrix + M

    D_control = sp.diags(p * control**(p - 1))
    drho_dcontrol = spsolve(H, (M @ D_control).tocsc())

    return np.asarray(dV_drho @ drho_dcontrol).ravel()