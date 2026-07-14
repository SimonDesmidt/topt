from sensitivities import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PolyCollection

## Plot FEM #####################################################################################################################

def extract_boundary_faces(elements):
    """Extract exterior triangular faces from tetrahedral connectivity."""
    elements = np.asarray(elements, dtype=np.int64)

    local_faces = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2]
    ], dtype=np.int64)

    faces = elements[:, local_faces].reshape(-1, 3)
    sorted_faces = np.sort(faces, axis=1)

    _, first_indices, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)

    return faces[first_indices[counts == 1]]


def camera_projection(coords, elev=25.0, azim=-60.0):
    """
    Orthographically project 3D coordinates onto a 2D camera plane.

    Returns
    -------
    projected : ndarray, shape (nNodes, 2)
        Projected coordinates.
    depth : ndarray, shape (nNodes,)
        Camera depth used to sort faces.
    """
    coords = np.asarray(coords, dtype=float)

    elev = np.deg2rad(elev)
    azim = np.deg2rad(azim)

    centered = coords - coords.mean(axis=0)

    camera_direction = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev)
    ])

    up = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(camera_direction, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])

    screen_x = np.cross(up, camera_direction)
    screen_x /= np.linalg.norm(screen_x)

    screen_y = np.cross(camera_direction, screen_x)
    screen_y /= np.linalg.norm(screen_y)

    projected = np.column_stack((centered @ screen_x, centered @ screen_y))
    depth = centered @ camera_direction

    return projected, depth


def add_projected_surface(ax, nodes, boundary_faces, values=None, cmap="hot", edgecolor="0.15", linewidth=0.15, alpha=1.0, elev=25.0, azim=-60.0):
    """
    Add a projected tetrahedral boundary surface to a standard 2D axis.

    Parameters
    ----------
    values : ndarray, optional
        Nodal scalar values. If provided, faces are colored using their mean
        nodal value.
    """
    projected, node_depth = camera_projection(nodes, elev=elev, azim=azim)

    polygons = projected[boundary_faces]
    face_depth = node_depth[boundary_faces].mean(axis=1)
    order = np.argsort(face_depth)

    polygons = polygons[order]

    if values is None:
        collection = PolyCollection(polygons, facecolors="none", edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    else:
        face_values = np.asarray(values)[boundary_faces].mean(axis=1)[order]
        collection = PolyCollection(polygons, array=face_values, cmap=cmap, edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)

    ax.add_collection(collection)

    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-12)
    padding = 0.04 * span

    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_aspect("equal")
    ax.set_axis_off()

    return collection

def plot_fem(mesh, p=3, scale=5000, elev=25.0, azim=-60.0):
    print("=" * 40)
    print("3D MBB BEAM FEM SOLVER")
    print("=" * 40 + "\n")

    density = np.ones(len(mesh.nodes), dtype=float)

    numNodes = len(mesh.nodes)
    numElements = len(mesh.elements)

    print(f"Created mesh with {numNodes} nodes and {numElements} tetrahedral elements.\n")

    # Consistent scalar mass matrix divided by tetrahedron volume:
    # M_e = V/20 * mass_pattern
    local_mass_matrix = np.array([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 2.0]
    ]) / 20.0

    # Compute total gravity force.
    f_gravity = np.zeros(3 * numNodes, dtype=float)

    for element_id, element in enumerate(mesh.elements):
        volume = mesh.volumes[element_id]
        fe = -ms.rho_matter * ms.gravity * volume * (local_mass_matrix @ density[element])
        np.add.at(f_gravity, 3 * element + 2, fe)

    total_gravity = -f_gravity[2::3].sum()

    # Integrate nodal density over the tetrahedral mesh.
    material_volume = 0.0

    for element_id, element in enumerate(mesh.elements):
        material_volume += mesh.volumes[element_id] * np.mean(density[element])

    theoretical_weight = ms.rho_matter * ms.gravity * material_volume

    print(f"  Total gravity force: {total_gravity:.2f} N")
    print(f"  Theoretical weight:  {theoretical_weight:.2f} N")
    print(f"  Difference:          {abs(total_gravity - theoretical_weight):.2e} N\n")

    # Solve the 3D FEM system.
    print("Solved FEM system:")

    u, K, rhs = fem_solver(mesh, density, p)

    u_x = u[0::3]
    u_y = u[1::3]
    u_z = u[2::3]
    u_mag = np.sqrt(u_x**2 + u_y**2 + u_z**2)

    print(f"  Max displacement: {np.max(u_mag) * 1000:.6f} mm")
    print(f"  Max |u_x|:        {np.max(np.abs(u_x)) * 1000:.6f} mm")
    print(f"  Max |u_y|:        {np.max(np.abs(u_y)) * 1000:.6f} mm")
    print(f"  Max |u_z|:        {np.max(np.abs(u_z)) * 1000:.6f} mm")

    compliance = u @ K @ u

    print(f"\n  Compliance: {compliance:.6f} J\n")

    node_coords = np.asarray(mesh.nodes, dtype=float)
    boundary_faces = extract_boundary_faces(mesh.elements)

    deformed_coords = node_coords.copy()
    deformed_coords[:, 0] += scale * u_x
    deformed_coords[:, 1] += scale * u_y
    deformed_coords[:, 2] += scale * u_z

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Displacement magnitude
    ax = axes[0]

    displacement_plot = add_projected_surface(
        ax,
        node_coords,
        boundary_faces,
        values=u_mag * 1000.0,
        cmap="hot",
        edgecolor="0.1",
        linewidth=0.15,
        elev=elev,
        azim=azim
    )

    colorbar = fig.colorbar(displacement_plot, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Displacement magnitude [mm]")

    ax.set_title("Displacement magnitude on exterior surface")

    # 2. Undeformed and deformed shapes
    ax = axes[1]
    add_projected_surface(ax, node_coords, boundary_faces, values=None, edgecolor="purple", linewidth=0.35, alpha=0.35, elev=elev, azim=azim)
    add_projected_surface(ax, deformed_coords, boundary_faces, values=None, edgecolor="cyan", linewidth=0.45, alpha=0.9, elev=elev, azim=azim)

    ax.set_title(f"Undeformed and deformed beam, displacement scale x{scale}")
    
    dirichlet_nodes = np.unique([condition[0] for condition in mesh.dirichlet])
    dirichlet_coords = mesh.nodes[dirichlet_nodes]
    ax.scatter(dirichlet_coords[:, 0], dirichlet_coords[:, 1], dirichlet_coords[:, 2], color="red")
    
    neumann_nodes = np.unique([condition[0] for condition in mesh.neumann])
    neumann_coords = mesh.nodes[neumann_nodes]
    ax.scatter(neumann_coords[:, 0], neumann_coords[:, 1], neumann_coords[:, 2], color="green")


    # Manual legend because PolyCollection wireframes do not create useful labels.
    undeformed_line = plt.Line2D([0], [0], color="purple", linewidth=1.5, alpha=0.35, label="Undeformed")
    deformed_line = plt.Line2D([0], [0], color="cyan", linewidth=1.5, alpha=0.9, label=f"Deformed (x{scale})")
    ax.legend(handles=[undeformed_line, deformed_line], loc="upper right")

    plt.tight_layout()
    plt.savefig("benchmark/mbb_beam_fem_3d.png", dpi=150, bbox_inches="tight")
    plt.show()
    
## Finite differences comparison ################################################################################################

def finite_differences(mesh, p, eps=1e-3):
    numNodes = len(mesh.nodes)
    print("="*40)
    print(f"SENSITIVITY ANALYSIS ({numNodes} nodes)")
    print("="*40 + "\n")

    np.random.seed(42)
    control = np.random.rand(len(mesh.nodes))
    r = mesh.r 
    M_filter = mesh.mass_matrix
    H_operator = r*r*mesh.stiff_matrix + M_filter

    ## Sensitivity of compliance w.r.t. control variable 
    print("Computing the gradient of the compliance: ")
    gradient = sensitivity_compliance(mesh, control, p)

    finite_diff_c = np.zeros(len(mesh.nodes))
    rel_err_vec = np.zeros(len(finite_diff_c))
    density_compliance = spsolve(H_operator, M_filter@control)
    for i in range(len(finite_diff_c)):
        control_p = control.copy()
        control_p[i] += eps
        density_p = spsolve(H_operator, M_filter@control_p**p)
        up, Kp, rhs_p = fem_solver(mesh, density_p, p)
        compliance_p = np.dot(Kp@up, up)

        control_m = control.copy()
        control_m[i] -= eps
        density_m = spsolve(H_operator, M_filter@control_m**p)
        um, Km, rhs_m = fem_solver(mesh, density_m, p)
        compliance_m = np.dot(Km@um, um)

        finite_diff_c[i] = (compliance_p - compliance_m) / (2*eps)
        rel_err_vec[i] = abs(gradient[i]-finite_diff_c[i])/np.maximum(abs(finite_diff_c[i]), eps)
        # print(f"{gradient[i]},  {finite_diff_c[i]}, {rel_err_vec[i]}")
    
    rel_err = np.linalg.norm(gradient-finite_diff_c) / np.maximum(np.linalg.norm(finite_diff_c), eps)

    print(f"  Relative error for the compliance sensitivity w.r.t. the control variable: {rel_err:.6e}\n")

    ## Sensitivity of compliance w.r.t. density variable 
    print("Computing the gradient of the compliance: ")
    
    r = mesh.r
    M = mesh.mass_matrix 
    H = r*r * mesh.stiff_matrix + M
    density_rho = spsolve(H, M@control**p)
    u, K, _ = fem_solver(mesh, density_rho, p)
    lmbda = -2*u # Adjoint parameter 

    drho_dk = spsolve(H, p*M@np.diag(control**(p-1)))
    df_drho = get_drhs_drho(mesh) # Derivative of the rhs term 
    dK_drho_u = get_dKu_drho(mesh, u, density_rho, p) # Derivative of the stiffness matrix K multiplied by the displacement vector u : dK_drho @ u
    gradient_c = ((u+lmbda).T @ dK_drho_u - lmbda.T @ df_drho)

    dc_drho = np.zeros(len(mesh.nodes))
    rel_err_vec_crho = np.zeros(len(dc_drho))
    for i in range(len(dc_drho)):
        density_p = density_rho.copy()
        density_p[i] += eps
        upp, Kp, rhs_p = fem_solver(mesh, density_p, p)
        compliance_p = np.dot(rhs_p, upp)

        density_m = density_rho.copy()
        density_m[i] -= eps
        umm, Km, rhs_m = fem_solver(mesh, density_m, p)
        compliance_m = np.dot(rhs_m, umm)

        dc_drho[i] = (compliance_p - compliance_m) / (2*eps)
        rel_err_vec_crho[i] = abs(gradient_c[i]-dc_drho[i])/abs(dc_drho[i])
    
    rel_err_crho = np.linalg.norm(gradient_c-dc_drho) / np.maximum(np.linalg.norm(dc_drho), 1e-10)

    print(f"  Relative error for the compliance sensitivity w.r.t. the density variable: {rel_err_crho:.6e}\n")

    ## Sensitivity of rhs w.r.t. density variable 
    print("Computing the gradient of the rhs:")

    gradient_rhs = get_drhs_drho(mesh)

    df_drho2 = np.zeros((3 * numNodes, numNodes))
    density_rhs = np.random.rand(numNodes)

    for i in range(numNodes):
        density_p = density_rhs.copy()
        density_p[i] += eps

        _, _, f_p = fem_solver(mesh, density_p, p)

        density_m = density_rhs.copy()
        density_m[i] -= eps

        _, _, f_m = fem_solver(mesh, density_m, p)

        df_drho2[:, i] = (f_p - f_m) / (2 * eps)    

    difference_rhs = gradient_rhs.toarray() - df_drho2
    rel_err_rhs_matrix = np.abs(difference_rhs) / np.maximum(np.abs(df_drho2), 1e-10)
    rel_err_rhs = np.linalg.norm(difference_rhs) / max(np.linalg.norm(df_drho2), 1e-14)

    print(f"  Relative error for rhs sensitivity w.r.t. density: {rel_err_rhs:.6e}\n")
    
    ## Sensitivity of Ku w.r.t. density variable 
    print("Computing the gradient of Ku:")

    dKu_drho = np.zeros((3 * numNodes, numNodes))

    density_Ku = 0.1 + 0.8 * np.random.rand(numNodes)
    u_Ku, _, _ = fem_solver(mesh, density_Ku, p)

    gradient_Ku = get_dKu_drho(mesh, u_Ku, density_Ku, p).toarray()

    for i in range(numNodes):
        density_p = density_Ku.copy()
        density_p[i] += eps
        _, Kp, _ = fem_solver(mesh, density_p, p)

        density_m = density_Ku.copy()
        density_m[i] -= eps
        _, Km, _ = fem_solver(mesh, density_m, p)

        dKu_drho[:, i] = (Kp @ u_Ku - Km @ u_Ku) / (2 * eps)

    difference_Ku = gradient_Ku - dKu_drho
    rel_err_Ku_matrix = np.abs(difference_Ku) / np.maximum(np.abs(dKu_drho), 1e-10)
    rel_err_Ku = np.linalg.norm(difference_Ku) / max(np.linalg.norm(dKu_drho), 1e-14)

    print(f"  Relative error for Ku sensitivity w.r.t. density: {rel_err_Ku:.6e}\n")

    ## Sensitivity of the density variable with respect to the control variable
    print("Computing drho/dk:")

    control_rho = 0.1 + 0.8 * np.random.rand(numNodes)

    D_control = sp.diags(p * control_rho**(p - 1))
    gradient_density = spsolve(H_operator.tocsc(), (M_filter @ D_control).tocsc()).toarray()

    drho_dk = np.zeros((numNodes, numNodes))
    for i in range(numNodes):
        control_p = control_rho.copy()
        control_p[i] += eps
        rho_p = spsolve(H_operator, M_filter @ control_p**p)

        control_m = control_rho.copy()
        control_m[i] -= eps
        rho_m = spsolve(H_operator, M_filter @ control_m**p)

        drho_dk[:, i] = (rho_p - rho_m) / (2 * eps)

    difference_density = gradient_density - drho_dk
    rel_err_density_matrix = np.abs(difference_density) / np.maximum(np.abs(drho_dk), 1e-10)
    rel_err_density = np.linalg.norm(difference_density) / max(np.linalg.norm(drho_dk), 1e-14)

    print(f"  Relative error for density sensitivity w.r.t. control: {rel_err_density:.6e}\n")

    # Create a single figure with all plots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Compliance sensitivity relative error
    ax1 = plt.subplot2grid((4, 1), (0,0), colspan=1)
    ax1.plot(rel_err_vec, color="red")
    ax1.set_title("Relative error of the compliance sensitivity w.r.t. the control variable")
    ax1.set_yscale("log")
    ax1.set_xlabel("Density index")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2-4: RHS sensitivity
    ax2 = plt.subplot(4, 3, 4)
    im2 = ax2.matshow(np.abs(gradient_rhs.toarray().T)+1e-12, cmap="hot", norm=LogNorm())
    ax2.set_title(r"Analytical $\partial f/\partial \rho$")
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ax3 = plt.subplot(4, 3, 5)
    im3 = ax3.matshow(np.abs(df_drho.toarray().T)+1e-12, cmap="hot", norm=LogNorm())
    ax3.set_title(r"Finite differences $\partial f/\partial \rho$")
    plt.colorbar(im3, ax=ax3)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = plt.subplot(4, 3, 6)
    im4 = ax4.matshow(rel_err_rhs_matrix.T+1e-12, cmap="hot", norm=LogNorm())
    ax4.set_title(r"$\partial f/\partial \rho$ - relative error" + "\n" + rf"(norm={rel_err_rhs:.2e})")
    plt.colorbar(im4, ax=ax4)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Plots 5-7: Ku sensitivity
    ax5 = plt.subplot(4, 3, 7)
    im5 = ax5.matshow(np.abs(gradient_Ku.T)+1e-12, cmap="hot", norm=LogNorm())
    ax5.set_title(r"Analytical $\partial K/\partial \rho \cdot u$")
    plt.colorbar(im5, ax=ax5)
    ax5.set_xticks([])
    ax5.set_yticks([])
    
    ax6 = plt.subplot(4, 3, 8)
    im6 = ax6.matshow(np.abs(dKu_drho.T)+1e-12, cmap="hot", norm=LogNorm())
    ax6.set_title(r"Finite differences $\partial K/\partial \rho \cdot u$")
    plt.colorbar(im6, ax=ax6)
    ax6.set_xticks([])
    ax6.set_yticks([])
    
    ax7 = plt.subplot(4, 3, 9)
    im7 = ax7.matshow(np.abs(rel_err_Ku_matrix.T)+1e-12, cmap="hot", norm=LogNorm())
    ax7.set_title(r"$\partial K/\partial \rho \cdot u$ - relative error" + "\n" + rf"(norm={rel_err_Ku:.2e})")
    plt.colorbar(im7, ax=ax7)
    ax7.set_xticks([])
    ax7.set_yticks([])
    
    # Plots 8-10: Density sensitivity 
    ax8 = plt.subplot(4, 3, 10)
    im8 = ax8.matshow(np.abs(gradient_density.T)+1e-12, cmap="hot", norm=LogNorm())
    ax8.set_title(r"Analytical $\partial \rho/\partial k$")
    plt.colorbar(im8, ax=ax8)
    ax8.set_xticks([])
    ax8.set_yticks([])
    
    ax9 = plt.subplot(4, 3, 11)
    im9 = ax9.matshow(np.abs(drho_dk.T)+1e-12, cmap="hot", norm=LogNorm())
    ax9.set_title(r"Finite differences $\partial \rho/\partial k$")
    plt.colorbar(im9, ax=ax9)
    ax9.set_xticks([])
    ax9.set_yticks([])
    
    ax10 = plt.subplot(4, 3, 12)
    im10 = ax10.matshow(np.abs(rel_err_density_matrix.T)+1e-12, cmap="hot", norm=LogNorm())
    ax10.set_title(rf"$\partial \rho/\partial k$ - relative error" + "\n" + rf"(norm={rel_err_density:.2e})")
    plt.colorbar(im10, ax=ax10)
    ax10.set_xticks([])
    ax10.set_yticks([])
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("benchmark/sensitivity_compliance.png", dpi=150, bbox_inches='tight')


def finite_differences_volume(mesh, alpha, p=3, eps=1e-3):
    """Check volume-constraint sensitivities on a tetrahedral mesh."""
    print("Computing the volume constraint sensitivity:\n")

    np.random.seed(42)
    num_nodes = len(mesh.nodes)

    control = 0.1 + 0.8 * np.random.rand(num_nodes)
    gradient = sensitivity_volume(mesh, control, p)

    finite_diff = np.zeros_like(gradient)

    for i in range(num_nodes):
        control_p = control.copy()
        control_p[i] += eps

        control_m = control.copy()
        control_m[i] -= eps

        constraint_p = volume_constraint(mesh, control_p, alpha, p)
        constraint_m = volume_constraint(mesh, control_m, alpha, p)

        finite_diff[i] = (constraint_p - constraint_m) / (2.0 * eps)

    difference = gradient - finite_diff
    rel_err_vec = np.abs(difference) / np.maximum(np.abs(finite_diff), 1e-10)
    rel_err = np.linalg.norm(difference) / max(np.linalg.norm(finite_diff), 1e-14)

    print(f"  Relative error for volume constraint w.r.t. control: {rel_err:.6e}\n")

    # Analytical derivative of material volume with respect to nodal density.
    gradient_rho = np.zeros(num_nodes)

    for element_id, element in enumerate(mesh.elements):
        gradient_rho[element] += mesh.volumes[element_id] / 4.0

    density = 0.1 + 0.8 * np.random.rand(num_nodes)
    finite_diff_rho = np.zeros_like(gradient_rho)

    def volume_rho(density_in):
        """Material volume for a prescribed nodal density field."""
        element_density = np.mean(density_in[mesh.elements], axis=1)
        return np.dot(mesh.volumes, element_density)

    for i in range(num_nodes):
        density_p = density.copy()
        density_p[i] += eps

        density_m = density.copy()
        density_m[i] -= eps

        volume_p = volume_rho(density_p)
        volume_m = volume_rho(density_m)

        finite_diff_rho[i] = (volume_p - volume_m) / (2.0 * eps)

    difference_rho = gradient_rho - finite_diff_rho
    err_rho_vec = np.abs(difference_rho) / np.maximum(np.abs(finite_diff_rho), 1e-10)
    err_rho = np.linalg.norm(difference_rho) / max(np.linalg.norm(finite_diff_rho), 1e-14)

    print(f"  Relative error for volume constraint w.r.t. density: {err_rho:.6e}\n")

    fig, ax = plt.subplots()
    ax.plot(rel_err_vec, label=r"$\mathrm{d}g/\mathrm{d}k$")
    ax.plot(err_rho_vec, label=r"$\mathrm{d}g/\mathrm{d}\rho$")
    ax.set_title("Relative error of the volume sensitivity")
    ax.set_xlabel("Nodal variable index")
    ax.set_ylabel("Relative error")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("benchmark/sensitivity_volume.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


## Run code #####################################################################################################################

if __name__ == "__main__":
    # Geometry of the beam 
    width = 3.0
    height = 1.0
    alpha = 0.5 # Maximum volume proportion
    eps = 1e-3
    p = 3

    # Load on the middle of the beam (upper left corner on the mesh)
    F = -4000*9.81   # Vertical load (N) at upper left (Liebherr R938)

    h = 0.3  # Maximum element size
    mesh = ms.create_mbb_mesh(width, height, h, F)

    plot_fem(mesh)
    finite_differences(mesh, p, eps)
    finite_differences_volume(mesh, alpha)

    plt.show()