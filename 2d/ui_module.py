import io
import os
import gmsh
import imageio
import numpy as np
from PIL import Image
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

class DensityView:
    """
    Single class for all visualisation needs:

    Live gmsh display
    -----------------
    * Density field as a continuous NodeData view (grey colormap, fixed [0,1]).
    * Optional front-node overlay: coloured spheres on the deforming mesh,
      enabled by passing node_tags to the constructor and calling
      update(..., front=...).

    Offline video export
    --------------------
    save_mp4(path)  — encodes all recorded frames to mp4 via imageio/ffmpeg.

    Parameters
    ----------
    node_tags : (N,) int
        gmsh 1-based tags aligned with the density / position arrays.
    elements  : (E, 3) int
        Triangle connectivity (0-based), used for Agg rendering.
    figsize : (w, h)
        Matplotlib figure size for exported frames.
    dpi : int
        Resolution of exported frames.
    """

    def __init__(self, node_tags, elements, figsize=(9, 3.5), dpi=150):
        if not gmsh.isInitialized():
            gmsh.initialize()

        self._tags = np.asarray(node_tags, dtype=np.int64)
        self._elements = np.asarray(elements,  dtype=int)
        self._figsize = figsize
        self._dpi = dpi
        self._step = 0
        self._frames: list[dict] = []

        # ── density view ─────────────────────────────────────────────────
        self._density_view = gmsh.view.add("density")
        idx = gmsh.view.get_index(self._density_view)
        gmsh.option.set_number(f"View[{idx}].IntervalsType", 3)   # continuous
        gmsh.option.set_number(f"View[{idx}].RangeType", 2)   # fixed range
        gmsh.option.set_number(f"View[{idx}].CustomMin", 0.0)
        gmsh.option.set_number(f"View[{idx}].CustomMax", 1.0)
        gmsh.option.set_number(f"View[{idx}].ShowScale", 1)
        gmsh.option.set_number(f"View[{idx}].ShowElement", 0)

        gmsh.fltk.initialize()

    def update(self, nodes, density, iteration, compliance):
        """
        Push one snapshot to the gmsh view(s) and record it for export.

        Parameters
        ----------
        nodes      : (N, 2|3) float   current node coordinates
        density    : (N,)     float   values in [0, 1]
        iteration  : int              iteration number (label)
        compliance : float            current compliance (label)
        """
        label = f"Iteration {iteration:4d}:   compliance = {compliance:.3f}Nm"

        # ── density NodeData ──────────────────────────────────────────────
        gmsh.view.addHomogeneousModelData(self._density_view, self._step, gmsh.model.getCurrent(), "NodeData", self._tags, density.astype(float))
        self._step += 1

        gmsh.graphics.draw()
        gmsh.fltk.awake("update")

        self._frames.append({"nodes": np.array(nodes, dtype=float), "density": np.array(density, dtype=float), "label": label})

    def _render_frame_rgb(self, frame: dict):
        """
        Render one frame to an (H, W, 3) uint8 array with headless Agg.
        Layout is controlled by self._layout.
        """
        coords = frame["nodes"][:, :2]
        density = frame["density"]
        label = frame["label"]

        triang = mtri.Triangulation(coords[:, 0], coords[:, 1], self._elements)
        level = np.linspace(0, 1, 5)
        levels = level[:-1]
        
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        ax.set_facecolor("0.9")
        cont = ax.tricontour(triang, density, levels=levels, cmap="jet", linewidths=1.2)
        ax.tricontourf(triang, density, levels=level, colors=["1.0", "0.85", "0.25", "0.15"])
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
        ax.set_title(label, fontsize=8, fontfamily="monospace", pad=3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"))
        buf.close()
        return arr

    def save_mp4(self, path, fps=10):
        """
        Encode all recorded frames to an mp4 file using imageio/ffmpeg.
        No display context or gmsh window is required.

        Parameters
        ----------
        path : output file path, e.g. "outputs/topopt.mp4"
        fps  : frames per second
        """
        if not self._frames:
            print("[DensityView] No frames recorded — mp4 not saved.")
            return

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        print(f"[DensityView] Rendering {len(self._frames)} frames...")
        arrays = [self._render_frame_rgb(f) for f in self._frames]

        # Pad to multiple of 16 (h264 macro-block requirement)
        h, w = arrays[0].shape[:2]
        h16 = int(np.ceil(h / 16)) * 16
        w16 = int(np.ceil(w / 16)) * 16
        if h16 != h or w16 != w:
            padded = []
            for a in arrays:
                p = np.full((h16, w16, 3), 255, dtype=np.uint8)
                p[:a.shape[0], :a.shape[1]] = a
                padded.append(p)
            arrays = padded

        writer = imageio.get_writer(path, format="ffmpeg", fps=fps, codec="libx264", output_params=["-pix_fmt", "yuv420p", "-crf", "18"])
        for arr in arrays:
            writer.append_data(arr)
        writer.close()
        print(f"[DensityView] mp4 saved to{path!r}  ({len(arrays)} frames @ {fps} fps)")