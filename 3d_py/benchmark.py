from sensitivities import * 
import matplotlib.tri as tri
from time import perf_counter
from density_3d import *
import matplotlib.pyplot as plt 
from scipy.sparse.linalg import splu
from matplotlib.animation import FuncAnimation

def fd_compliance(mesh, p, eps=1e-6):
    """ 
    Compute the gradient of the compliance w.r.t. control variable using finite differences and adjoint method.
    """
    np.random.seed(42)
    n = len(mesh.nodes)
    control = np.random.rand(n)

    r = mesh.r
    M = mesh.mass_matrix
    H = r * r * mesh.stiff_matrix + M
    
    t0 = perf_counter()
    gradient = sensitivity_compliance(mesh, control, p)
    t1 = perf_counter()
    time_am = t1 - t0
    
    t0 = perf_counter()
    H_lu = splu(H.tocsc())
    finite_diff = np.zeros(len(mesh.nodes))
    
    def compliance(ctrl):
        rhs = M @ (ctrl ** p)
        density = H_lu.solve(rhs)
        u, K, _ = fem_solver(mesh, density, p)
        return np.dot(K @ u, u)

    for i in range(len(mesh.nodes)):
        control_p = control.copy()
        control_p[i] += eps
        Jp = compliance(control_p)
        
        control_m = control.copy()
        control_m[i] -= eps
        Jm = compliance(control_m)
        finite_diff[i] = (Jp - Jm) / (2.0 * eps)

    t1 = perf_counter()
    time_fd = t1 - t0

    return time_fd, time_am, np.linalg.norm(gradient-finite_diff) / np.linalg.norm(finite_diff)

def complexity_gradient():
    """ 
    Complexity graph of the gradient computation, through adjoint method and finite differences. 
    """
    refinements = np.array([0.3,0.25,0.2,0.15,0.1,0.09,0.08,0.07,0.06,0.05])
    p=3
    fd_times = np.zeros(len(refinements))
    am_times = np.zeros(len(refinements))
    numNodes = np.zeros(len(refinements))
    errors   = np.zeros(len(refinements))
    for i, h in enumerate(refinements):
        mesh = ms.create_mbb_mesh(3.0, 1.0, h, -4000*9.81)
        numNodes[i] = len(mesh.elements)
        fd_times[i], am_times[i], errors[i] = fd_compliance(mesh, p)
        print(f"Done for h={h}: n={int(numNodes[i])}")
    np.savetxt("outputs/time_complexity.csv", np.array([numNodes, fd_times, am_times, errors]).T, delimiter=",")
    
    numNodes, fd_times, am_times, errors = np.loadtxt("outputs/time_complexity.csv", delimiter=",").T
    
    coeffs = np.polyfit(np.log(numNodes), np.log(fd_times), 1)
    fd_exp, fdp = coeffs
    coeffs = np.polyfit(np.log(numNodes), np.log(am_times), 1)
    am_exp, amp = coeffs
    
    plt.figure()
    plt.scatter(numNodes, fd_times, label="Finite differences", marker="s", color="royalblue")
    plt.plot(numNodes, np.exp(fdp)*numNodes**fd_exp, label=fr"$\mathcal{{O}}(n^{{{np.round(fd_exp,2)}}})$", linestyle="--", color="royalblue")
    plt.scatter(numNodes, am_times, label="Adjoint method", marker="o", color="crimson")
    plt.plot(numNodes, np.exp(amp)*numNodes**am_exp, label=fr"$\mathcal{{O}}(n^{{{np.round(am_exp,2)}}})$", linestyle="--", color="crimson")
    plt.grid()
    plt.xlabel("Number of nodes")
    plt.ylabel("Time [s]")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("outputs/time_complexity.png")
    plt.savefig("../prod/images/density/time_complexity_adjoint.pdf")
    
    plt.figure()
    plt.plot(numNodes, errors)
    plt.title("Relative error of the sensitivities")
    plt.grid()
    plt.xlabel("Number of nodes")
    plt.ylabel("Relative error")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("outputs/sensitivity_err.png")
    plt.savefig("../prod/images/density/sensitivity_err.pdf")
    plt.show()
    
def young_density():
    """ 
    Plot of the Young modulus for different p and the HS physical bound. 
    """
    nu = 0.3
    E_0 = 1e-6
    E_1 = 1.0
    
    x = np.linspace(0.0, 1.0, 100)
    colors = ["red", "cyan", "magenta"]
    plt.figure()
    for i, p in enumerate([1,2,3]):
        y = E_0 + (E_1 - E_0) * x**p 
        plt.plot(x, y, color=colors[i], label=f"SIMP - p={p}")
    hs_upper = np.where(x < 1e-10, E_0, E_1 * x / (1 + (1 - x) * (1 + nu) / (2 * (1 - 2 * nu))))
    plt.plot(x, hs_upper, color="orange", label=rf"Hashin-Shtrikman bound")

    plt.axhline(E_0, color="k", linestyle="--")
    plt.axhline(E_1, color="k", linestyle="-.")
    plt.yticks([E_0, E_1], [r"$E_{void}$", r"$E_{matter}$"])
    plt.xlabel(r"Density $\rho$")
    plt.ylabel(r"Effective Young modulus $E(\rho)/E_0$")
    plt.title(r"$E(\rho) = E_{void} + (E_{matter} - E_{void}) \rho^p$", fontsize=16)
    plt.legend()
    plt.savefig("outputs/density.png")
    plt.savefig("../prod/images/density/young_density.pdf")
    plt.show()

def density_repartition():
    h = 0.02
    mesh = ms.create_mbb_mesh(3., 1., h, -4000 * 9.81)
    volfrac = 0.25
    density, _, _, _, _ = density_approach(mesh, volfrac, plot=False)
    
    alpha = np.round(np.dot(mesh.areas, np.mean(density[mesh.elements], axis=1)) / np.sum(mesh.areas),2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(np.arange(len(density)), density, c=density, cmap="berlin", vmin=0, vmax=1, marker="x", alpha=0.8)
    ax1.grid(ls="--", lw=0.5, alpha=0.7)
    ax1.set_xlabel("Node index", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.axhline(0.5, linestyle="--", color="k", linewidth=1.2, label="Threshold = 0.5")
    ax1.legend(fontsize=8, loc="upper right")

    bins = np.linspace(0, 1, 11) 
    counts, _ = np.histogram(density, bins=bins)
    proportions = counts / len(density)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    colors = plt.cm.berlin(bin_centers)
    ax2.barh(bin_centers, proportions, height=0.08, color=colors, edgecolor="black", linewidth=0.5)

    ax2.set_xlabel("Proportion", fontsize=10)
    ax2.grid(axis="y", ls="--", lw=0.5, alpha=0.7)
    ax2.set_xlim(0, max(proportions)*1.25)

    for y, p in zip(bin_centers, proportions):
        ax2.text(p + 0.01, y, f"{p*100:.2f}%", ha="left", va="center", fontsize=8, fontweight="bold")

    plt.suptitle(rf"Nodal density distribution for $\alpha=${alpha}")
    plt.tight_layout()
    plt.savefig("outputs/density_values.png", dpi=200)
    plt.savefig("../prod/images/density/density_repartition.pdf", dpi=200)
    plt.show()

def convergence():
    h = 0.03
    mesh = ms.create_mbb_mesh(3., 1., h, -4000 * 9.81)
    volfrac = 0.25
    _, _, _, _, logs = density_approach(mesh, volfrac, init_design=0.5*np.ones(len(mesh.nodes)), log=True, plot=False)
    compliances, volumes, densities, variations = logs
    threshold = variations[-1]
    triang = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)

    snapshot_iters = [1, 3, 5, 6, 8, len(densities)-1]
    cmap = plt.get_cmap("jet")
    snapshot_colors = cmap(np.linspace(0.0, 1.0, len(snapshot_iters)))

    iters_c = np.arange(len(compliances))
    iters_v = np.arange(len(variations))
    level = np.linspace(0, 1, 5, endpoint=True)
    levels = level[:-1]

    fig = plt.figure(figsize=(7, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.18, 0.7, 0.55])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])
    ax = [ax0, ax1, ax2]

    # ------------------------------------------------------------
    # Plot 1: zoomed compliance and volume, first iterations
    # ------------------------------------------------------------
    n_zoom = 11
    ax[0].plot(iters_c[:n_zoom], compliances[:n_zoom], color="crimson", linewidth=1.8, label="Compliance")
    ax[0].set_ylabel("Compliance [Nm]", fontsize=10)
    ax[0].set_yscale("log")
    ax[0].grid(ls="--", lw=0.5, alpha=0.7)
    ax[0].set_title("Convergence of the compliance", fontsize=11, pad=8)
    ax[0].set_xlim(0, n_zoom-1)
    ax[0].set_ylim(min(compliances)*0.9, max(compliances)*1.1)

    ax0b = ax[0].twinx()
    ax0b.plot(iters_c[:n_zoom], np.array(volumes[:n_zoom])*100, color="darkgreen", linewidth=1.8, label="Volume")
    ax0b.axhline(volfrac*100, color="k", linewidth=0.5, linestyle="--")
    ax0b.set_ylabel("Volume fraction [%]")
    ax0b.set_ylim(24, 26)

    ax[0].axhline(min(compliances), label="Minimum of the compliance", color="crimson", linestyle="--")
    lines1, labels1 = ax[0].get_legend_handles_labels()
    lines2, labels2 = ax0b.get_legend_handles_labels()
    ax[0].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    for it, color in zip(snapshot_iters, snapshot_colors):
        if it < n_zoom:
            ax[0].axvline(it, color=color, linestyle="dashed", linewidth=2.0, alpha=0.8)

    # ------------------------------------------------------------
    # Insets around the zoomed plot: first five snapshots
    # flipped top/bottom compared with previous version
    # ------------------------------------------------------------
    y_bottom, y_top = ax[0].get_ylim()
    snapshot_iters_zoom = [it for it in snapshot_iters[:-1] if it < n_zoom]
    snapshot_colors_zoom = snapshot_colors[:len(snapshot_iters_zoom)]

    for j, (it, color) in enumerate(zip(snapshot_iters_zoom, snapshot_colors_zoom)):
        inset_width = 0.28
        inset_height = 0.28
        x_inset = np.clip((j // 2) / 3, 0.02, 1.0 - inset_width - 0.02)
        y_inset = 1.08 if j % 2 == 0 else -0.42
        y_anchor = y_top if j % 2 == 0 else y_bottom

        inset = ax[0].inset_axes([x_inset, y_inset, inset_width, inset_height], transform=ax[0].transAxes)
        inset.set_facecolor("0.15")
        inset.tricontourf(triang, densities[it], levels=level, colors=["1.0", "0.85", "0.25", "0.15"])

        inset.tricontour(triang, densities[it], levels=levels, cmap="jet", linewidths=1.2)
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])

        x_center = x_inset + inset_width/2.0
        y_center = y_inset + inset_height/2.0
        y_center += -inset_height/4.0 if j % 2 == 0 else inset_height/4.0

        ax[0].annotate("", xy=(it, y_anchor), xycoords=ax[0].transData, xytext=(x_center, y_center), textcoords=ax[0].transAxes, arrowprops=dict(arrowstyle="-", linestyle="dashed", linewidth=2.0, color=color, alpha=0.8), annotation_clip=False)

    # ------------------------------------------------------------
    # Plot 2: full compliance and volume history
    # ------------------------------------------------------------
    ax[1].plot(iters_c, compliances, color="crimson", linewidth=1.8, label="Compliance")
    ax[1].set_ylabel("Compliance [Nm]", fontsize=10)
    ax[1].set_yscale("log")
    ax[1].grid(ls="--", lw=0.5, alpha=0.7)

    ax1b = ax[1].twinx()
    ax1b.plot(iters_c, np.array(volumes)*100, color="darkgreen", linewidth=1.8, label="Volume")
    ax1b.set_yticks([24, 24.5, 25, 25.5, 26])
    ax1b.axhline(volfrac*100, color="k", linewidth=0.5, linestyle="--")
    ax1b.set_ylabel("Volume fraction [%]")
    ax1b.set_ylim(24, 26)

    lines1, labels1 = ax[1].get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    for it, color in zip(snapshot_iters, snapshot_colors):
        ax[1].axvline(it, color=color, linestyle="dashed", linewidth=2.0, alpha=0.8)

    # ------------------------------------------------------------
    # Sixth/final inset: placed on upper zoom plot, connected from lower full-history plot
    # ------------------------------------------------------------
    it = snapshot_iters[-1]
    color = snapshot_colors[-1]
    
    inset_width = 0.28
    inset_height = 0.28
    x_inset = np.clip(1, 0.02, 1.0 - inset_width - 0.02)
    y_inset = -0.42
    y_anchor = y_bottom

    inset = ax[0].inset_axes([x_inset, y_inset, inset_width, inset_height], transform=ax[0].transAxes)
    inset.set_facecolor("0.15")
    inset.tricontourf(triang,densities[it],levels=level, colors=["1.0", "0.85", "0.25", "0.15"])
    inset.tricontour(triang, densities[it], levels=levels, cmap="jet", linewidths=1.2)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    ax[1].axvline(it, color=color, linestyle="dashed", linewidth=2.0, alpha=0.8)

    y_anchor = ax[1].get_ylim()[1]
    x_center = x_inset + inset_width / 2.0
    y_center = y_inset + inset_height / 5.5

    ax[1].annotate("", xy=(it, y_anchor), xycoords=ax[1].transData, xytext=(x_center, y_center), textcoords=ax[0].transAxes, arrowprops=dict(arrowstyle="-", linestyle="dashed", linewidth=2.0, color=color, alpha=0.8), annotation_clip=False)

    # ------------------------------------------------------------
    # Plot 3: relative design variation, smaller height
    # ------------------------------------------------------------
    ax[2].plot(iters_v, variations, color="royalblue", linewidth=1.8, label=r"$\frac{\|\rho_{i+1}-\rho_i\|}{\|\rho_i\|}$")
    ax[2].set_xlabel("Iterations", fontsize=10)
    ax[2].set_ylabel("Relative variation", fontsize=10)
    ax[2].set_yscale("log")
    ax[2].grid(ls="--", lw=0.5, alpha=0.7)
    ax[2].axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax[2].text(len(variations)*0.02, threshold*1.2, "threshold", fontsize=8)
    ax[2].legend(loc="upper right", fontsize=10)

    plt.savefig("outputs/compliance_history_to.png", dpi=200, bbox_inches="tight")
    plt.savefig("../prod/images/density/compliance_history_topt.pdf", dpi=200, bbox_inches="tight")
    plt.show()

def interface(duration=10):
    h = 0.02
    mesh = ms.create_mbb_mesh(3., 1., h, -4000 * 9.81)
    # mesh = ms.create_connecting_rod_mesh()
    # mesh = ms.create_crane_mesh(width=30, height=20, mast_width=2.5, beam_height=1.5, triangle_height=4.0, mesh_size=0.15, F=-4000*9.81)
    volfrac = 0.25

    _, _, _, _, logs = density_approach(mesh, volfrac, log=True, plot=False)
    _, _, densities, _ = logs

    triang = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    fig, ax = plt.subplots(figsize=(8,6))
    level = np.linspace(0, 1, 5)
    levels = level[:-1]

    ax.set_facecolor("0.9")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        for artist in ax.collections[:]:
            artist.remove()

        ax.set_title(f"Iteration {i}")
        ax.tricontour(triang, densities[i], levels=levels, cmap="jet", linewidths=1.2)
        ax.tricontourf(triang, densities[i], levels=level, colors=["1.0", "0.85", "0.25", "0.15"])
        
        return ax.collections

    ani = FuncAnimation(fig, update, frames=len(densities), interval=50, blit=False)
    ani.save("outputs/interface.mp4", writer="ffmpeg", fps=len(densities)//duration)
    # ani.save("outputs/crane2.mp4", writer="ffmpeg", fps=len(densities)//duration)

    for artist in ax.collections[:]:
        artist.remove()

    cont = ax.tricontour(triang, densities[-1], levels=levels, cmap="jet", linewidths=1.2)
    ax.tricontourf(triang, densities[-1], levels=level, colors=["1.0", "0.85", "0.25", "0.15"])
    
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])

    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks(level)
    cbar_ax.set_xlabel("Density")
    
    for lvl, color in zip(levels, cont.cvalues):
        cbar_ax.axvline(lvl, color=cont.cmap(cont.norm(color)), linewidth=2.0)
    cbar_ax.axvline(1.0,color=cont.cmap(1.0),linewidth=2.0)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")

    plt.savefig("outputs/isovalues.png", bbox_inches="tight")
    plt.savefig("../prod/images/density/isovalues.pdf", bbox_inches="tight")
    # plt.savefig("outputs/crane.png", bbox_inches="tight")
    # plt.savefig("../prod/images/density/crane.pdf", bbox_inches="tight")
    plt.show()
    
def radius_impact():
    h = 0.02
    mesh = ms.create_mbb_mesh(3., 1., h, -4000 * 9.81)
    volfrac = 0.25
    radii = [0, h/np.sqrt(12), h, 5*h]
    titles = [r"$r/h=0$", r"$r/h=1/\sqrt{12}$", r"$r/h=1$", r"$r/h=2$"]

    level = np.linspace(0, 1, 5)
    levels = level[:-1]
    
    fig, axes = plt.subplots(2,2, figsize=(12,6))
    for r, title, ax in zip(radii, titles, axes.ravel()) : 
        density, _, _, _, _ = density_approach(mesh, volfrac, radius=r, log=False, plot=False)

        triang = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)

        ax.set_aspect("equal")
        ax.set_facecolor("0.15")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(title)
        cont = ax.tricontour(triang,density,levels=levels,cmap="jet",linewidths=1.2)
        fill = ax.tricontourf(triang,density,levels=level,colors=["1.0", "0.85", "0.25", "0.15"])
        
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])

    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks(level)
    cbar_ax.set_xlabel("Density")

    for lvl, color in zip(levels, cont.cvalues):
        cbar_ax.axvline(lvl, color=cont.cmap(cont.norm(color)), linewidth=2.0)
    cbar_ax.axvline(1.0,color=cont.cmap(1.0),linewidth=2.0)

    plt.savefig("outputs/radius_impact.png", bbox_inches="tight")
    plt.savefig("../prod/images/density/radius_impact.pdf", bbox_inches="tight")
    plt.show()
    
def refinement_impact():
    refinements = [0.07,0.05,0.03,0.02]
    fig, axes = plt.subplots(2,2, figsize=(12,6))
    
    level = np.linspace(0, 1, 5)
    levels = level[:-1]
    
    for h, ax in zip(refinements, axes.ravel()) : 
        mesh = ms.create_mbb_mesh(3., 1., h, -4000 * 9.81)
        volfrac = 0.25
        density, _, _, _, logs = density_approach(mesh, volfrac, log=False, plot=False)

        triang = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)

        ax.set_aspect("equal")
        ax.set_facecolor("0.15")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f"{len(mesh.nodes)} nodes")
        cont = ax.tricontour(triang,density,levels=levels,cmap="jet",linewidths=1.2)
        fill = ax.tricontourf(triang,density,levels=level,colors=["1.0", "0.85", "0.25", "0.15"])
        print(f"Done for h={h}")
        
    # To add a zoom
    axins = axes[1,1].inset_axes([0.85,0.5,0.12,0.4], xlim=(1.8,1.95), ylim=(0.6, 0.75))
    axins.set_facecolor("0.15")
    axins.tricontour(triang, density, levels=levels, cmap="jet", linewidths=1.2)
    axins.tricontourf(triang, density, levels=level, colors=["1.0", "0.85", "0.25", "0.15"])
    axins.set_xticks([])
    axins.set_yticks([])
    axes[1,1].indicate_inset_zoom(axins, edgecolor="black")
        
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks(level)
    cbar_ax.set_xlabel("Density")
    
    for lvl, color in zip(levels, cont.cvalues):
        cbar_ax.axvline(lvl, color=cont.cmap(cont.norm(color)), linewidth=2.0)
    cbar_ax.axvline(1.0,color=cont.cmap(1.0),linewidth=2.0)

    plt.savefig("outputs/refinement_impact.png", bbox_inches="tight")
    plt.savefig("../prod/images/density/refinement_impact.pdf", bbox_inches="tight")
    plt.show()

def complexity_topt():
    refinements = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])

    times = np.zeros(len(refinements))
    nodes = np.zeros(len(refinements))
    iters = np.zeros(len(refinements))

    for i, h in enumerate(refinements):
        mesh = ms.create_mbb_mesh(3.0, 1.0, h, -4000 * 9.81)
        nodes[i] = len(mesh.nodes)

        start = perf_counter()
        _, it, _, _, _ = density_approach(mesh, 0.25, plot=False)
        end = perf_counter()
        times[i] = end - start

        iters[i] = it
        print(f"Done for h={h}, {int(nodes[i])} nodes")

    np.savetxt("outputs/complexity_topt.csv", np.array([nodes, times, iters]).T)

    nodes, times, iters = np.loadtxt("outputs/complexity_topt.csv").T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    coeffs = np.polyfit(np.log(nodes), np.log(times), 1)
    a1, p1 = coeffs
    ax1.scatter(nodes, times, marker="o", color="royalblue", label="Topology optimisation")
    ax1.plot(nodes, np.exp(p1)*nodes**a1, linestyle="--", color="royalblue", label=rf"$\mathcal{{O}}(n^{{{a1:.2f}}})$")

    coeffs = np.polyfit(np.log(nodes), np.log(times/iters), 1)
    a2, p2 = coeffs
    ax1.scatter(nodes, times/iters,marker="o", color="crimson", label="Average per iteration")
    ax1.plot(nodes, np.exp(p2)*nodes**a2, linestyle="--", color="crimson", label=rf"$\mathcal{{O}}(n^{{{a2:.2f}}})$")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of nodes")
    ax1.set_ylabel("Time [s]")
    ax1.grid()
    ax1.legend()

    ax2.plot(nodes, iters, marker="o", color="black")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of nodes")
    ax2.set_ylabel("Iterations")
    ax2.grid()

    plt.tight_layout()
    plt.savefig("outputs/complexity_topt.png", bbox_inches="tight")
    plt.savefig("../prod/images/density/complexity_topt.pdf", bbox_inches="tight")
    plt.show()

def iterations():
    h = 0.03
    mesh = ms.create_mbb_mesh(3., 1., h, -4000 * 9.81)
    volfrac = 0.25

    _, _, _, _, logs = density_approach(mesh, volfrac, log=True, plot=False)
    compliances, volumes, densities, _ = logs
    print("Done")

    triang = tri.Triangulation(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    level = np.linspace(0, 1, 5)
    levels = level[:-1]
    cmap_lines = plt.get_cmap("jet")
    
    fig, axes = plt.subplots(3,2, figsize=(12,8))
    iter_values = [1,3,5,6,8,len(densities)-1]
    for it, ax in zip(iter_values, axes.ravel()):
        ax.set_facecolor("0.15")
        ax.tricontourf(triang, densities[it], levels=level, colors=["1.0", "0.85", "0.25", "0.15"])
        cont = ax.tricontour(triang, densities[it], levels=levels, cmap="jet", linewidths=1.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if it == len(densities)-1:
            ax.set_title(rf"Iteration {it} (final) - Compliance = {compliances[it]:.3f}Nm")# + "\n" + f"Volume fraction = {volumes[it]*100:.1f}%")
        else : 
            ax.set_title(rf"Iteration {it} - Compliance = {compliances[it]:.3f}Nm")# + "\n" + f"Volume fraction = {volumes[it]*100:.1f}%")
    
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])

    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks(level)
    cbar_ax.set_xlabel("Density")
    
    for lvl, color in zip(levels, cont.cvalues):
        cbar_ax.axvline(lvl, color=cont.cmap(cont.norm(color)), linewidth=2.0)
    cbar_ax.axvline(1.0,color=cont.cmap(1.0),linewidth=2.0)
    
    plt.savefig("outputs/iterations.png", bbox_inches="tight")
    plt.savefig("../prod/images/density/iterations.pdf",bbox_inches="tight")
    plt.show()
            
if __name__ == "__main__":
    # complexity_gradient()
    
    # young_density()
    
    # density_repartition()
    
    # convergence()

    interface()
    
    # radius_impact()
    
    # complexity_topt()
    
    # refinement_impact()
    
    # iterations()