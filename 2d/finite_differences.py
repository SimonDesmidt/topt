from sensitivities import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

## Plot FEM #####################################################################################################################
def plot_fem(mesh, scale=500):
    print("="*40)
    print("MBB BEAM FEM SOLVER")
    print("="*40 + "\n")

    density = np.ones(len(mesh.nodes))
    
    numNodes = len(mesh.nodes)
    numElements = len(mesh.elements)
    print(f"Created mesh with {numNodes} nodes and {numElements} elements.\n")
    
    # Compute total gravity force
    f_gravity = np.zeros(2*numNodes)
    local_mass_matrix = 1/24 * np.array([[2,1,1], [1,2,1], [1,1,2]])
    for element_id, element in enumerate(mesh.elements):
        jac = mesh.areas[element_id] * 2
        fe = -ms.rho_matter * ms.gravity * jac * local_mass_matrix @ density[element] 
        for i in range(3):
            f_gravity[2*element[i]+1] += fe[i]
    
    total_gravity = -f_gravity[1::2].sum()  # Sum y-components (negative because downward)

    volume = 0.0
    for id, element in enumerate(mesh.elements) :
        volume += mesh.areas[id] * (np.sum(density[element]))/3
    theoretical_weight = ms.rho_matter * ms.gravity * volume

    print(f"  Total gravity force: {total_gravity:.2f} N")
    print(f"  Theoretical weight:  {theoretical_weight:.2f} N")
    print(f"  Difference: {abs(total_gravity - theoretical_weight):.2e} N\n")
    
    # Solve system
    print("Solved FEM system:")
    u, K, rhs = fem_solver(mesh, density, p)
    print(f"  Max displacement: {np.max(np.abs(u))*1000:.6f} mm")
    print(f"  Max u_x: {np.max(np.abs(u[::2]))*1000:.6f} mm")
    print(f"  Max u_y: {np.max(np.abs(u[1::2]))*1000:.6f} mm")
    
    # Compute compliance
    compliance = u @ K @ u
    print(f"\n  Compliance: {compliance:.6f} J\n")

    node_coords = mesh.nodes 
    elements = mesh.elements 

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    u_x = u[::2]
    u_y = u[1::2]
    u_mag = np.sqrt(u_x**2 + u_y**2)
    
    # 1. Displacement magnitude
    ax = axes[0]
    triplot = ax.tripcolor(node_coords[:, 0], node_coords[:, 1], elements, u_mag * 1000, shading='flat', cmap="hot")
    ax.triplot(node_coords[:, 0], node_coords[:, 1], elements, 'k-', lw=0.3, alpha=0.3)
    plt.colorbar(triplot, ax=ax, label='Displacement magnitude [mm]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Displacement Magnitude')
    ax.set_aspect('equal')
    
    # 2. Deformed shape
    ax = axes[1]
    deformed_coords = node_coords.copy()
    deformed_coords[:, 0] += scale * u_x
    deformed_coords[:, 1] += scale * u_y
    ax.triplot(node_coords[:, 0], node_coords[:, 1], elements, 'b-', lw=0.5, alpha=0.3, label='Undeformed')
    ax.triplot(deformed_coords[:, 0], deformed_coords[:, 1], elements, 'r-', lw=0.5,label=f'Deformed (x{scale})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Deformed beam')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('outputs/mbb_beam_fem.png', dpi=150, bbox_inches='tight')

## Finite differences comparison ################################################################################################

def finite_differences(mesh, p, eps=1e-3):
    print("="*40)
    print("SENSITIVITY ANALYSIS")
    print("="*40 + "\n")
    numNodes = len(mesh.nodes)

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
    
    density_rho = spsolve(H_operator, M_filter@control**p)
    u_rho, K, rhs_rho = fem_solver(mesh, density_rho, p)
    lmbda = spsolve((K.T+K)/2, -rhs_rho) # Adjoint parameter 

    df_drho = get_drhs_drho(mesh) # Derivative of the rhs term 
    dK_drho_u = get_dKu_drho(mesh, u_rho, density_rho, p) # Derivative of the stiffness matrix K multiplied by the displacement vector u : dK_drho @ u

    gradient_c = u_rho.T @ df_drho + lmbda.T @ (dK_drho_u - df_drho)

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
    df_drho2 = np.zeros((2*numNodes, numNodes))
    density_rhs = np.random.rand(numNodes)
    for i in range(numNodes):
        density_p = density_rhs.copy()
        density_p[i] += eps 
        _, _, f_p = fem_solver(mesh, density_p, p)

        density_m = density_rhs.copy()
        density_m[i] -= eps
        _, _, f_m = fem_solver(mesh, density_m, p)

        df_drho2[:,i] = (f_p-f_m)/(2*eps)
    
    rel_err_rhs = np.abs(gradient_rhs - df_drho2) / np.maximum(np.abs(df_drho2), 1e-10)
    print(f"  Relative error for the rhs sensitivity w.r.t. the density:                 {np.linalg.norm(rel_err_rhs):.6e}\n")

    ## Sensitivity of Ku w.r.t. density variable 
    print("Computing the gradient of Ku:")
    dKu_drho = np.zeros((2*numNodes, numNodes))
    density_Ku = np.random.rand(numNodes)
    u_Ku, _, _ = fem_solver(mesh, density_Ku, p)
    gradient_Ku = get_dKu_drho(mesh, u_Ku, density_Ku, p).toarray()
    for i in range(numNodes):
        density_p = density_Ku.copy()
        density_p[i] += eps 
        _, Kp, _ = fem_solver(mesh, density_p, p)

        density_m = density_Ku.copy()
        density_m[i] -= eps 
        _, Km, _ = fem_solver(mesh, density_m, p)

        dKu_drho[:,i] = (Kp@u_Ku - Km@u_Ku)/(2*eps)
    rel_err_Ku = np.abs(gradient_Ku - dKu_drho) / np.maximum(np.abs(dKu_drho), 1e-10)
    print(f"  Relative error for the Ku sensitivity w.r.t. the density:                  {np.linalg.norm(rel_err_Ku):.6e}\n")

    ## Sensitivity of the density variable w.r.t the control variable
    print("Computing drho/dk:")
    control_rho = np.random.rand(numNodes)
    gradient_density = spsolve(H_operator.tocsc(), M_filter.tocsc()@np.diag(control_rho**(p-1))*p)
    drho_dk = np.zeros((numNodes, numNodes))
    for i in range(numNodes):
        control_p = control_rho.copy()
        control_p[i] += eps 
        rho_p = spsolve(H_operator, M_filter@control_p**p)

        control_m = control_rho.copy()
        control_m[i] -= eps 
        rho_m = spsolve(H_operator, M_filter@control_m**p)

        drho_dk[:,i] = (rho_p-rho_m)/(2*eps)
    rel_err_density = np.abs(gradient_density - drho_dk) / np.maximum(np.abs(drho_dk), 1e-10)
    print(f"  Relative error for the density sensitivity w.r.t the control variable:     {np.linalg.norm(rel_err_density):.6e}\n")

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
    im4 = ax4.matshow(rel_err_rhs.T+1e-12, cmap="hot")
    ax4.set_title(r"$\partial f/\partial \rho$ - relative error" + "\n" + rf"(norm={np.linalg.norm(rel_err_rhs):.2e})")
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
    im7 = ax7.matshow(np.abs(rel_err_Ku.T)+1e-12, cmap="hot", norm=LogNorm())
    ax7.set_title(r"$\partial K/\partial \rho \cdot u$ - relative error" + "\n" + rf"(norm={np.linalg.norm(rel_err_Ku):.2e})")
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
    im10 = ax10.matshow(np.abs(rel_err_density.T)+1e-12, cmap="hot", norm=LogNorm())
    ax10.set_title(rf"$\partial \rho/\partial k$ - relative error" + "\n" + rf"(norm={np.linalg.norm(rel_err_density):.2e})")
    plt.colorbar(im10, ax=ax10)
    ax10.set_xticks([])
    ax10.set_yticks([])
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("outputs/sensitivity_compliance.png", dpi=150, bbox_inches='tight')
    plt.savefig("../prod/images/density/sensitivity_compliance.pdf", dpi=150, bbox_inches='tight')

    fig, ax = plt.subplots()
    triplot = ax.tripcolor(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements, rel_err_vec, shading='flat', cmap="hot")
    ax.triplot(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements, 'k-', lw=0.3, alpha=0.3)
    plt.colorbar(triplot, ax=ax, label='Relative error')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Relative error of the compliance sensitivity')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_beam.png")


def finite_differences_volume(mesh, alpha, p=3, eps=1e-3):
    print("Computing the volume constraint sensitivity: \n")
    np.random.seed(42)
    control = np.random.rand(len(mesh.nodes))
    gradient = sensitivity_volume(mesh, control, p)

    finite_diff = np.zeros_like(gradient)
    rel_err_vec = np.zeros_like(gradient)
    for i in range(len(finite_diff)):
        control_p = control.copy()
        control_p[i] += eps 

        control_m = control.copy()
        control_m[i] -= eps 
        
        finite_diff[i] = (volume_constraint(mesh, control_p, alpha, p) - volume_constraint(mesh, control_m, alpha, p)) / (2*eps)
        rel_err_vec[i] = abs(gradient[i]-finite_diff[i]) / np.maximum(abs(finite_diff[i]), 1e-10)

    rel_err = np.abs(gradient - finite_diff) / np.maximum(np.abs(finite_diff), 1e-10)
    print(f"  Relative error for the volume constraint w.r.t. the control variable:      {np.linalg.norm(rel_err):.6e}\n")

    gradient_rho = np.zeros(len(mesh.nodes))
    for element_id, element in enumerate(mesh.elements):
        gradient_rho[element] += mesh.areas[element_id] / 3.0

    finite_diff_rho = np.zeros_like(gradient_rho)
    err_rho_vec = np.zeros_like(gradient_rho)
    density = np.random.rand(len(mesh.nodes))

    def volume_rho(density_in):
        volume = 0.0
        for element_id, element in enumerate(mesh.elements):
            volume += mesh.areas[element_id] * np.sum(density_in[element]) / 3.0
            volume += mesh.areas[element_id]
        return volume
    
    for i in range(len(finite_diff_rho)):
        density_p = density.copy()
        density_p[i] += eps 

        density_m = density.copy()
        density_m[i] -= eps 

        finite_diff_rho[i] = (volume_rho(density_p) - volume_rho(density_m)) / (2*eps)
        err_rho_vec[i] = abs(gradient_rho[i] - finite_diff_rho[i]) / np.maximum(abs(finite_diff_rho[i]), 1e-10)
    
    err_rho = np.linalg.norm(gradient_rho - finite_diff_rho) / np.maximum(np.linalg.norm(finite_diff_rho), 1e-10)
    print(f"  Relative error for the volume constraint w.r.t. the density variable:      {np.linalg.norm(err_rho):.6e}\n")

    plt.figure()
    plt.plot(rel_err_vec, label=r"$d/dk$")
    plt.plot(err_rho_vec, label=r"$d/d\rho$")
    plt.title("Relative error of the volume sensitivity")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/sensitivity_volume.png")

def finite_diff_evolution(mesh_size, eps=1e-3):
    print("="*40)
    print("SENSITIVITY ANALYSIS WITH FINE MESH")
    print("="*40 + "\n")
    
    error = np.zeros(len(mesh_size))
    numElems = np.zeros(len(mesh_size))
    np.random.seed(42)
    print("Computing the gradient of the compliance w.r.t. the control variable for several meshes: \n")
    for iter, size in enumerate(mesh_size) :
        mesh = ms.create_mbb_mesh(3., 1., size, -40000*9.81)
        numNodes = len(mesh.nodes)
        control = np.random.rand(numNodes)
        r = mesh.r 
        M_filter = mesh.mass_matrix
        H_operator = r*r*mesh.stiff_matrix + M_filter

        ## Sensitivity of compliance w.r.t. control variable 
        gradient = sensitivity_compliance(mesh, control, p)

        finite_diff_c = np.zeros(len(mesh.nodes))
        rel_err_vec = np.zeros(len(finite_diff_c))

        for i in range(len(finite_diff_c)):
            control_p = control.copy()
            control_p[i] += eps
            density_p = spsolve(H_operator, M_filter@control_p)
            up, Kp, _ = fem_solver(mesh, density_p, p)
            compliance_p = np.dot(Kp@up, up)

            control_m = control.copy()
            control_m[i] -= eps
            density_m = spsolve(H_operator, M_filter@control_m)
            um, Km, _ = fem_solver(mesh, density_m, p)
            compliance_m = np.dot(Km@um, um)

            finite_diff_c[i] = (compliance_p - compliance_m) / (2*eps)
            rel_err_vec[i] = abs(gradient[i]-finite_diff_c[i])/np.maximum(abs(finite_diff_c[i]), eps)
        
        error[iter] = np.linalg.norm(gradient-finite_diff_c) / np.maximum(np.linalg.norm(finite_diff_c), eps)
        numElems[iter] = len(mesh.elements)
        print(f"  Relative error for a mesh of {int(numElems[iter])} elements: {error[iter]:.6e}\n")
    
    plt.figure()
    plt.plot(numElems, error, marker="o", linewidth=2, color="red")
    plt.grid()
    plt.xlabel("Number of elements in the mesh")
    plt.ylabel("Relative error of the sensitivity")
    plt.yscale("log")
    plt.title("Evolution of the relative error of the sensitivity")
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_refinement.png")


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

    h = 0.2  # Maximum element size
    mesh = ms.create_mbb_mesh(width, height, h, F)

    # plot_fem(mesh)
    finite_differences(mesh, p, eps)
    # finite_differences_volume(mesh, alpha)
    # finite_diff_evolution([0.3,0.25,0.2,0.15,0.1,0.08])

    plt.show()