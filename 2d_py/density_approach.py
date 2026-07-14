import gmsh
import numpy as np
import ui_module as ui 
import mesh_struct as ms
from scipy.sparse import coo_matrix, bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized

def barycentric_integral(density_element, p):
    """
    Vectorised barycentric integral of the Young modulus over all elements.
    density_element : (nel, 3) nodal densities per element
    Returns         : (nel,)   effective Young modulus per element

    Closed-form expansion of sum_{i+j+k=p} r0^i r1^j r2^k / ((p+1)(p+2))
    for p in {1, 2, 3}; falls back to a for loop for higher values.
    """
    r0, r1, r2 = density_element[:,0], density_element[:,1], density_element[:,2]
    p = int(p)
    if p == 1:
        bary = (r0 + r1 + r2) / 6.0
    elif p == 2:
        bary = (r0**2 + r1**2 + r2**2 + r0*r1 + r0*r2 + r1*r2) / 12.0
    elif p == 3:
        bary = (r0**3 + r1**3 + r2**3
                + r0**2*r1 + r0**2*r2
                + r1**2*r0 + r1**2*r2
                + r2**2*r0 + r2**2*r1
                + r0*r1*r2) / 20.0
    else:
        def _scalar(rv):
            b = 0
            for i in range(p+1):
                for j in range(p+1-i):
                    ell = p-i-j
                    if ell >= 0:
                        b += rv[0]**i * rv[1]**j * rv[2]**ell
            return b / ((p+1)*(p+2))
        bary = np.array([_scalar(row) for row in density_element])
    return (ms.E_void + (ms.E_matter - ms.E_void) * bary)*0.5

def fem_solver(mesh, density, p):
    numNodes = len(mesh.nodes)
    local_mass_matrix = 1/24 * np.array([[2,1,1], [1,2,1], [1,1,2]])

    density_el = density[mesh.elements]
    integral_of_Young = barycentric_integral(density_el, p)
    sK = (mesh.Ke.reshape(-1,36) * integral_of_Young[:,None]).ravel()

    K = coo_matrix((sK, (mesh.iK, mesh.jK)), shape=(2*numNodes, 2*numNodes)).tocsr()

    # gravity
    fe = density_el @ local_mass_matrix
    fe *= (-ms.rho_matter * ms.gravity * 2*mesh.areas[:,None])

    rhs_y = np.zeros(numNodes)
    np.add.at(rhs_y, mesh.elements, fe)

    rhs = np.zeros(2*numNodes)
    rhs[1::2] = rhs_y

    # Neumann BC, xy or normal/tangent
    for condition in mesh.neumann:
        if len(condition) == 3:
            node_id, fx, fy = condition
            rhs[2 * node_id] += fx
            rhs[2 * node_id + 1] += fy

        if len(condition) == 4:
            node_id, a, b, frame = condition

            if frame == "xy":
                rhs[2 * node_id] += a
                rhs[2 * node_id + 1] += b

            if frame == "nt":
                n = np.asarray(mesh.boundary_normals[node_id], dtype=float)
                t = np.asarray(mesh.boundary_tangents[node_id], dtype=float)
                if np.linalg.norm(n) < 1e-14: raise ValueError(f"No valid normal found at node {node_id}.")

                n = n / np.linalg.norm(n)
                t = np.array([-n[1], n[0]], dtype=float) if np.linalg.norm(t) < 1e-14 else t / np.linalg.norm(t)
                
                force = np.zeros(2)
                if a is not None: force += a * n
                if b is not None: force += b * t

                rhs[2 * node_id] += force[0]
                rhs[2 * node_id + 1] += force[1]

    # Dirichlet BC as general linear constraints Cdir u = values
    rows = []
    cols = []
    data = []
    values = []
    row = 0

    for condition in mesh.dirichlet:
        node_id, ux, uy = condition
        
        if ux is not None:
            rows.append(row); cols.append(2 * node_id); data.append(1.0); values.append(ux); row += 1
        if uy is not None:
            rows.append(row); cols.append(2 * node_id + 1); data.append(1.0); values.append(uy); row += 1

    Cdir = coo_matrix((data, (rows, cols)), shape=(row, 2*numNodes)).tocsr()
    values = np.asarray(values, dtype=float)

    if Cdir.shape[0] == 0:
        u = spsolve(K, rhs)
    else:
        zero = csr_matrix((Cdir.shape[0], Cdir.shape[0]))
        system = bmat([[K, Cdir.T], [Cdir, zero]], format="csr")
        rhs_aug = np.concatenate([rhs, values])
        sol_aug = spsolve(system, rhs_aug)
        u = sol_aug[:2 * numNodes]

    return u, K, rhs

##  Topology optimisation
def density_approach(mesh, volfrac, radius=None, p=3, max_iter=1000, init_design=None, plot=True, outfile=None, log=False):
    """Topology optimization on given mesh with triangular elements and Helmholtz filter"""
    threshold = 2e-3
    if plot: 
        print("Topology optimization with Helmholtz PDE filter")
        print(f"Elements: {len(mesh.elements)}, Nodes: {len(mesh.nodes)}")
        print(f"Volume fraction: {volfrac}, p: {p}")

    # Initialize design variables
    numNodes = len(mesh.nodes)
    design = volfrac * np.ones(numNodes) if init_design is None else init_design.copy() # design variable (unfiltered)
    old_design = design.copy()
    density = design.copy()
    vtot = np.sum(np.array(mesh.areas))

    M_filter = mesh.mass_matrix
    if radius is None : radius = mesh.r
    H_operator = radius*radius * mesh.stiff_matrix + M_filter
    H_solver = factorized(H_operator.tocsc())

    # gmsh view and logs
    view = ui.DensityView(mesh.node_tags, mesh.elements) if plot else None 
    compliances = []
    volume_constraints = []
    densities = []
    variations = []

    # Gradient of the constraint, constant 
    dv_phys = np.zeros(numNodes)
    for element_id, element in enumerate(mesh.elements):
        dv_phys[element] += mesh.areas[element_id] / 3
    dv = H_solver(M_filter@dv_phys)

    # Optimisation loop
    it = 0
    change = 1.0
    while change > threshold and it < max_iter:
        density = np.clip(H_solver(M_filter@design), 0.0, 1.0)
        u, K, rhs = fem_solver(mesh, density, p)
        
        # Objective and sensitivities
        ue = u[mesh.edofMat] # (nel, 6)
        ce_el = np.einsum('ei,eij,ej->e', ue, mesh.Ke, ue)
        ce = np.zeros(numNodes)
        np.add.at(ce, mesh.elements, ce_el[:, None])

        dc_phys = -p * density**(p - 1) * (ms.E_matter - ms.E_void) * ce
        dc = H_solver(M_filter @ dc_phys)

        # Optimality criteria update 
        old_design = design.copy()
        design = oc(design, dc, dv, volfrac * vtot)
        change = np.linalg.norm(design - old_design) / np.linalg.norm(old_design)

        compliance = np.dot(u, rhs)
        compliances.append(compliance)
        volume_constraints.append(np.sum(np.mean(density[mesh.elements], axis=1)*mesh.areas)/vtot)
        densities.append(density)
        variations.append(change)

        it += 1
        if plot : 
            print(f"  iteration {it:4d}  |  compliance = {compliance:.4e}  |  design variation = {change:.4f}")
            view.update(mesh.nodes, density, iteration=it, compliance=compliance)

    if plot:
        view.update(mesh.nodes, density, iteration=it, compliance=compliance)
        gmsh.fltk.run()
        if outfile is not None: 
            view.save_mp4(outfile, fps=5)
    
    log_data = [compliances, volume_constraints, densities, variations] if log else None 
    return density, it, u, rhs, log_data

def oc(x, dc, dv, vol_target):
    l1, l2 = 0.0, 1e9
    move = 0.2

    while (l2 - l1) / (l2 + 1e-12) > 1e-3:
        lmid = 0.5 * (l1 + l2)

        ratio = np.maximum(-dc / (dv * lmid), 1e-12)
        xnew = np.maximum(0.0, np.maximum(x - move,
                np.minimum(1.0, np.minimum(x + move, x*np.sqrt(ratio)))))

        if np.sum(dv * xnew) > vol_target:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


if __name__ == "__main__":
    gmsh.initialize()

    h = 0.03
    outfile = None
    # mesh = ms.create_mbb_mesh(3., 1., h, -4000 * 9.81)
    # outfile = "outputs/to.mp4"
    mesh = ms.create_connecting_rod_mesh(length=0.30, r_big_outer=0.045, r_big_inner=0.03, r_small_outer=0.028, r_small_inner=0.013, shank_width=0.032, mesh_size=0.001, F=-4000*9.81)
    outfile = "outputs/connecting_rod.mp4"
    # mesh = ms.create_crane_mesh(width=30, height=20, mast_width=2.5, beam_height=1.5, triangle_height=4.0, mesh_size=0.15, F=-4000*9.81)
    # outfile = "outputs/crane.mp4"
    # mesh = ms.create_bridge_mesh(width=100.0, height=5.0, mesh_size=0.4, total_load=-4000*9.81)
    # outfile = "outputs/bridge.mp4"
    # mesh = ms.create_wishbone_mesh(scale=0.15, mesh_size=0.002, F=-500*9.81)
    # outfile = "outputs/wishbone.mp4"
    volfrac = 0.25
    results, _, _, _, _ = density_approach(mesh, volfrac, p=3, outfile=outfile)

    gmsh.finalize()