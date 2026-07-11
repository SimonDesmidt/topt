import numpy as np
import mesh_struct as ms
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

## FEM solver ###################################################################################################################

def fem_solver(mesh, density, p):
    """ Compute and solve the linear system for linear elasticity.
    Returns:
     - u the displacement vector
     - K the stiffness matrix 
     - f the force vector (rhs)
    """
    
    nNodes = len(mesh.nodes)
    nElements = len(mesh.elements)
    
    # Each element contributes 6x6 = 36 entries
    nnz = 36 * nElements
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)
    
    rhs = np.zeros(2*nNodes)
    local_mass_matrix = 1/24 * np.array([[2,1,1], [1,2,1], [1,1,2]])
    
    integral_of_Young = np.array([mesh.areas[element_id] * (ms.E_void + (ms.E_matter - ms.E_void) * np.mean(density[element])) for element_id, element in enumerate(mesh.elements)]) #  E(rho) = E_v + (E_m-E_v)rho
    
    idx = 0
    for element_id, element in enumerate(mesh.elements):
        B = mesh.B_matrices[element_id]
        jac = mesh.areas[element_id] * 2
        
        Ke = integral_of_Young[element_id] * B.T @ ms.C @ B
        fe = -ms.rho_matter * ms.gravity * jac * local_mass_matrix @ density[element]
        
        dofs = np.empty(6, dtype=np.int32)
        dofs[0::2] = 2 * element      # x DOFs
        dofs[1::2] = 2 * element + 1  # y DOFs
        
        # Vectorized assembly using outer product for indices
        dof_rows, dof_cols = np.meshgrid(dofs, dofs, indexing='ij')
        
        rows[idx:idx+36] = dof_rows.ravel()
        cols[idx:idx+36] = dof_cols.ravel()
        vals[idx:idx+36] = Ke.ravel()
        idx += 36
        
        rhs[2*element+1] += fe # Force vector (only y component due to gravity)
    
    K = sp.coo_matrix((vals, (rows, cols)), shape=(2*nNodes, 2*nNodes)).tolil()
    
    # Apply boundary conditions
    for c in mesh.dirichlet:
        id = 2*c[0]
        if c[1] is not None: 
            K[id, :] = 0.
            K[id, id] = 1.
            rhs[id] = c[1]
        if c[2] is not None: 
            K[id+1, :] = 0.
            K[id+1, id+1] = 1.
            rhs[id+1] = c[2]
    
    for c in mesh.neumann: 
        id = 2*c[0]
        rhs[id] += c[1]
        rhs[id+1] += c[2]
    
    K = K.tocsr()
    u = spsolve(K, rhs)
    
    return u, K, rhs

## Sensitivity computation ######################################################################################################

def get_drhs_drho(mesh):
    """ Get the derivative of the rhs of the FEM system. """
    numNodes = len(mesh.nodes)
    df_drho = sp.lil_matrix((2*numNodes, numNodes))
    parent_mass = 1/24 * np.array([[2,1,1], [1,2,1], [1,1,2]])

    for element_id, element in enumerate(mesh.elements):
        jac = mesh.areas[element_id] * 2 
        for i, ni in enumerate(element):
            for j, nj in enumerate(element):
                df_drho[2*ni+1, nj] += parent_mass[i,j] * jac 
    
    df_drho *= -ms.rho_matter * ms.gravity 
    for c in mesh.dirichlet: # Apply the boundary conditions (neumann not necessary)
        if c[1] is not None:
            df_drho[2*c[0]] = 0.0
        if c[2] is not None:
            df_drho[2*c[0]+1] = 0.0
    return df_drho.tocsr()

def get_dKu_drho(mesh, u, density, p):
    numNodes = len(mesh.nodes)
    dK_drho_u = sp.lil_matrix((2*numNodes, numNodes))
    for element_id, element in enumerate(mesh.elements):
        Be = mesh.B_matrices[element_id]
        ue = np.zeros(6)
        for i, ni in enumerate(element):
            ue[2*i] = u[2*ni]
            ue[2*i+1] = u[2*ni+1]
        BTCBu = Be.T @ ms.C @ Be @ ue 
        int_dE_drho = (ms.E_matter - ms.E_void) / 3.0 * mesh.areas[element_id] * np.ones(3)
        dKe_drhoe_u = np.outer(BTCBu, int_dE_drho)
        for i, ni in enumerate(element):
            for j, nj in enumerate(element):
                dK_drho_u[2*ni, nj] += dKe_drhoe_u[2*i, j]
                dK_drho_u[2*ni+1, nj] += dKe_drhoe_u[2*i+1,j]
    
    for c in mesh.dirichlet: # Apply the boundary conditions (neumann not necessary)
        if c[1] is not None:
            dK_drho_u[2*c[0],:] = 0.0
        if c[2] is not None:
            dK_drho_u[2*c[0]+1,:] = 0.0
    
    return dK_drho_u.tocsr()

def sensitivity_compliance(mesh, control, p):
    """ Compute the sensitivity of the compliance with respect to the control variable. """
    r = mesh.r
    M = mesh.mass_matrix 
    H = r*r * mesh.stiff_matrix + M
    density = spsolve(H, M@control**p)
    u, K, _ = fem_solver(mesh, density, p)
    lmbda = spsolve(K.T, -(K+K.T)@u) # Adjoint parameter 

    drho_dk = spsolve(H, p*M@np.diag(control**(p-1)))
    df_drho = get_drhs_drho(mesh) # Derivative of the rhs term 
    df_dk = df_drho @ drho_dk
    dK_drho_u = get_dKu_drho(mesh, u, density, p) # Derivative of the stiffness matrix K multiplied by the displacement vector u : dK_drho @ u
    dK_dk_u = dK_drho_u @ drho_dk
    # This also works: 
    # lmbda = spsolve(K.T, -rhs)
    # dc_drho = u.T @ df_drho + lmbda.T @ (dK_drho_u - df_drho)
    return (u+lmbda).T @ dK_dk_u - lmbda.T @ df_dk

## Volume constraint and its gradient ###########################################################################################

def volume_constraint(mesh, control, alpha, p=3):
    """ Compute the value of the volume constraint : V(rho) - alpha * V_t <= 0. """
    volume = 0.0
    v_tot = 0.0
    r = mesh.r 
    M_filter = mesh.mass_matrix
    H_operator = r*r*mesh.stiff_matrix + M_filter
    density = spsolve(H_operator, M_filter@control**p)
    for element_id, element in enumerate(mesh.elements):
        volume += mesh.areas[element_id] * np.sum(density[element]) / 3.0
        v_tot += mesh.areas[element_id]

    return volume - alpha * v_tot 

def sensitivity_volume(mesh, control, p):
    """ Compute the gradient of the volume constraint w.r.t. the control variable: dV(rho)/drho * drho/dk """
    gradient = np.zeros(len(mesh.nodes))
    for element_id, element in enumerate(mesh.elements):
        gradient[element] += mesh.areas[element_id] / 3.0
    
    r = mesh.r 
    M_filter = mesh.mass_matrix
    H_operator = r*r*mesh.stiff_matrix + M_filter
    return gradient @ spsolve(H_operator, M_filter@np.diag(control**(p-1)*p))