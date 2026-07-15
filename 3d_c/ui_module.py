import io
import os
import shutil
import subprocess
import sys

import gmsh
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image


class DensityView:
    """Visualize nodal density fields on a linear tetrahedral mesh.

    The class supports:
    - a live continuous Gmsh NodeData view when Python Gmsh has FLTK support;
    - continuous or binary offline rendering without ``mpl_toolkits.mplot3d``;
    - MP4 export;
    - an interactive standalone Gmsh view of the material region ``rho >= iso_value``;
    - display of the initial rectangular design-domain edges.

    Parameters
    ----------
    node_tags : array_like, shape (n_nodes,)
        Gmsh node tags aligned with ``nodes`` and ``density``.
    elements : array_like, shape (n_elements, 4)
        Zero-based tetrahedral connectivity.
    figsize : tuple, optional
        Figure size used for rendered frames.
    view_elev, view_azim : float, optional
        Orthographic camera angles in degrees.
    show_domain_edges : bool, optional
        Draw the 12 edges of the initial rectangular design domain.
    interactive_mesh_path : str, optional
        Path used for the standalone Gmsh surface export.
    """

    _TETRA_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    _LOCAL_FACES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)

    def __init__(self, node_tags, elements, figsize=(9, 6), view_elev=25.0, view_azim=-60.0, show_domain_edges=True, interactive_mesh_path="outputs/topology_result.msh", COLOR=np.array([65, 105, 225]) / 255.0): #4169E1
        if not gmsh.isInitialized():
            gmsh.initialize()

        self._tags = np.asarray(node_tags, dtype=np.int64).reshape(-1)
        self._elements = np.asarray(elements, dtype=np.int64)
        self._figsize = figsize
        self._view_elev = float(view_elev)
        self._view_azim = float(view_azim)
        self._iso_value = 0.5
        self._show_domain_edges = bool(show_domain_edges)
        self._interactive_mesh_path = os.path.abspath(interactive_mesh_path)
        self._step = 0
        self._frames = []
        self._COLOR = COLOR

        if self._elements.ndim != 2 or self._elements.shape[1] != 4:
            raise ValueError("elements must have shape (n_elements, 4) for linear tetrahedra.")

        self._boundary_faces = self._extract_boundary_faces(self._elements)

        self._density_view = gmsh.view.add("density")
        view_index = gmsh.view.getIndex(self._density_view)
        gmsh.option.setNumber(f"View[{view_index}].IntervalsType", 3)
        gmsh.option.setNumber(f"View[{view_index}].RangeType", 2)
        gmsh.option.setNumber(f"View[{view_index}].CustomMin", 0.0)
        gmsh.option.setNumber(f"View[{view_index}].CustomMax", 1.0)
        gmsh.option.setNumber(f"View[{view_index}].ShowScale", 1)
        gmsh.option.setNumber(f"View[{view_index}].ShowElement", 0)

        if gmsh.fltk.isAvailable():
            gmsh.fltk.initialize()

    @staticmethod
    def _extract_boundary_faces(elements):
        """Return triangular faces belonging to exactly one tetrahedron."""
        faces = elements[:, DensityView._LOCAL_FACES].reshape(-1, 3)
        sorted_faces = np.sort(faces, axis=1)
        _, first_indices, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
        return faces[first_indices[counts == 1]]

    @staticmethod
    def _interpolate_iso_point(p0, p1, rho0, rho1, iso_value):
        """Interpolate the ``rho=iso_value`` point along one edge."""
        denominator = rho1 - rho0
        if abs(denominator) <= 1e-14:
            return 0.5 * (p0 + p1)
        t = np.clip((iso_value - rho0) / denominator, 0.0, 1.0)
        return p0 + t * (p1 - p0)

    @staticmethod
    def _sort_polygon_points(points):
        """Sort coplanar points around their centroid."""
        center = points.mean(axis=0)
        normal = None
        for i in range(1, len(points) - 1):
            candidate = np.cross(points[i] - points[0], points[i + 1] - points[0])
            norm = np.linalg.norm(candidate)
            if norm > 1e-14:
                normal = candidate / norm
                break
        if normal is None:
            return points
        axis_1 = points[0] - center
        if np.linalg.norm(axis_1) <= 1e-14:
            axis_1 = points[1] - center
        axis_1 /= np.linalg.norm(axis_1)
        axis_2 = np.cross(normal, axis_1)
        relative = points - center
        angles = np.arctan2(relative @ axis_2, relative @ axis_1)
        return points[np.argsort(angles)]

    def _extract_isosurface(self, coords, density, iso_value=None):
        """Extract the linearly interpolated isosurface with marching tetrahedra."""
        iso_value = self._iso_value if iso_value is None else float(iso_value)
        triangles = []

        for element in self._elements:
            points = coords[element]
            values = density[element]
            intersections = []

            for local_a, local_b in self._TETRA_EDGES:
                rho_a = values[local_a]
                rho_b = values[local_b]
                crosses = (rho_a < iso_value <= rho_b) or (rho_b < iso_value <= rho_a)
                if not crosses:
                    continue
                intersections.append(self._interpolate_iso_point(points[local_a], points[local_b], rho_a, rho_b, iso_value))

            unique_points = []
            for point in intersections:
                if not any(np.linalg.norm(point - existing) <= 1e-12 for existing in unique_points):
                    unique_points.append(point)

            if len(unique_points) == 3:
                triangles.append(np.asarray(unique_points))
            elif len(unique_points) == 4:
                polygon = self._sort_polygon_points(np.asarray(unique_points))
                triangles.append(np.array([polygon[0], polygon[1], polygon[2]]))
                triangles.append(np.array([polygon[0], polygon[2], polygon[3]]))

        return np.asarray(triangles, dtype=float) if triangles else np.empty((0, 3, 3), dtype=float)

    @staticmethod
    def _clip_triangle_above_iso(points, values, iso_value=0.5):
        """Clip one triangle by ``rho >= iso_value`` and triangulate the result."""
        polygon = [(points[i], values[i]) for i in range(3)]
        clipped = []

        for i in range(3):
            point_a, value_a = polygon[i]
            point_b, value_b = polygon[(i + 1) % 3]
            inside_a = value_a >= iso_value
            inside_b = value_b >= iso_value

            if inside_a: clipped.append(point_a)

            if inside_a != inside_b:
                clipped.append(DensityView._interpolate_iso_point(point_a, point_b, value_a, value_b, iso_value))

        if len(clipped) < 3:
            return []

        clipped = np.asarray(clipped, dtype=float)
        if len(clipped) == 3:
            return [clipped]
        if len(clipped) == 4:
            return [np.array([clipped[0], clipped[1], clipped[2]]), np.array([clipped[0], clipped[2], clipped[3]])]
        return []

    def _extract_material_boundary_faces(self, coords, density, iso_value=None):
        """Extract exterior design-domain faces retained by ``rho >= iso_value``."""
        iso_value = self._iso_value if iso_value is None else float(iso_value)
        triangles = []
        for face in self._boundary_faces:
            triangles.extend(self._clip_triangle_above_iso(coords[face], density[face], iso_value))
        return np.asarray(triangles, dtype=float) if triangles else np.empty((0, 3, 3), dtype=float)

    def _extract_matter_surface(self, coords, density, iso_value=None):
        """Build the closed boundary of the material region ``rho >= iso_value``."""
        iso_value = self._iso_value if iso_value is None else float(iso_value)
        internal_surface = self._extract_isosurface(coords, density, iso_value)
        exterior_surface = self._extract_material_boundary_faces(coords, density, iso_value)
        if len(internal_surface) == 0:
            return exterior_surface
        if len(exterior_surface) == 0:
            return internal_surface
        return np.concatenate((internal_surface, exterior_surface), axis=0)

    @staticmethod
    def _initial_box_edges(coords):
        """Return the eight corners and twelve edges of the bounding box."""
        xmin, ymin, zmin = coords.min(axis=0)
        xmax, ymax, zmax = coords.max(axis=0)
        corners = np.array([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin], [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]])
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)
        return corners, edges

    @staticmethod
    def _camera_basis(elev, azim):
        """Return screen axes and viewing direction for the orthographic camera."""
        elev = np.deg2rad(elev)
        azim = np.deg2rad(azim)
        camera_direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        global_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(camera_direction, global_up)) > 0.99:
            global_up = np.array([0.0, 1.0, 0.0])
        screen_x = np.cross(global_up, camera_direction)
        screen_x /= np.linalg.norm(screen_x)
        screen_y = np.cross(camera_direction, screen_x)
        screen_y /= np.linalg.norm(screen_y)
        return screen_x, screen_y, camera_direction

    @classmethod
    def _camera_projection(cls, coords, elev, azim, center=None):
        """Orthographically project 3D coordinates onto a 2D camera plane."""
        coords = np.asarray(coords, dtype=float)
        center = coords.mean(axis=0) if center is None else np.asarray(center, dtype=float)
        xyz = coords - center
        screen_x, screen_y, camera_direction = cls._camera_basis(elev, azim)
        projected = np.column_stack((xyz @ screen_x, xyz @ screen_y))
        depth = xyz @ camera_direction
        return projected, depth

    @staticmethod
    def _normalize_rows(vectors):
        """Normalize an array of vectors row by row."""
        norms = np.linalg.norm(vectors, axis=1)
        result = np.zeros_like(vectors)
        good = norms > 1e-14
        result[good] = vectors[good] / norms[good, None]
        return result

    def _binary_face_colors(self, triangles):
        """Compute shaded royal-blue RGBA colors for binary surface triangles."""
        normals = self._normalize_rows(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]))
        light_direction = np.array([0.35, -0.45, 0.82], dtype=float)
        light_direction /= np.linalg.norm(light_direction)
        diffuse = np.abs(normals @ light_direction)
        intensity = 0.30 + 0.65 * diffuse
        rgb = intensity[:, None] * self._COLOR[None, :]
        rgb = np.clip(rgb, 0.0, 1.0)
        return np.column_stack((rgb, np.ones(len(intensity))))

    @staticmethod
    def _merge_triangle_vertices(triangles, tolerance=1e-10):
        """Convert triangle soup to indexed vertices and triangular connectivity."""
        triangles = np.asarray(triangles, dtype=float)
        if len(triangles) == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

        scale = 1.0 / tolerance
        vertex_map = {}
        vertices = []
        faces = np.empty((len(triangles), 3), dtype=np.int64)

        for triangle_id, triangle in enumerate(triangles):
            for local_id, point in enumerate(triangle):
                key = tuple(np.round(point * scale).astype(np.int64))
                if key not in vertex_map:
                    vertex_map[key] = len(vertices)
                    vertices.append(point)
                faces[triangle_id, local_id] = vertex_map[key]

        return np.asarray(vertices, dtype=float), faces

    def update(self, nodes, density, iteration, compliance):
        """Push one snapshot to Gmsh and record it for rendering and export."""
        nodes = np.asarray(nodes, dtype=float)
        density = np.asarray(density, dtype=float).reshape(-1)

        if nodes.shape != (len(self._tags), 3):
            raise ValueError(f"nodes must have shape ({len(self._tags)}, 3).")
        if density.shape != (len(self._tags),):
            raise ValueError(f"density must have shape ({len(self._tags)},).")

        density = np.clip(density, 0.0, 1.0)
        label = f"Iteration {iteration:4d}: compliance = {compliance:.3f} Nm\n {len(nodes)} nodes"

        gmsh.view.addHomogeneousModelData(self._density_view, self._step, gmsh.model.getCurrent(), "NodeData", self._tags, density)
        self._step += 1

        if gmsh.fltk.isAvailable():
            gmsh.graphics.draw()
            gmsh.fltk.awake("update")

        self._frames.append({"nodes": nodes.copy(), "density": density.copy(), "label": label})

    def _draw_domain_edges(self, ax, coords, center):
        """Overlay projected initial-domain edges on a Matplotlib axis."""
        if not self._show_domain_edges:
            return
        corners, edges = self._initial_box_edges(coords)
        projected, _ = self._camera_projection(corners, self._view_elev, self._view_azim, center=center)
        for edge in edges:
            edge_points = projected[edge]
            ax.plot(edge_points[:, 0], edge_points[:, 1], color="0.05", linewidth=1.0, alpha=0.85, zorder=10)

    def _render_frame_rgb(self, frame):
        """Render one recorded 3D frame to an RGB array without mplot3d."""
        coords = np.asarray(frame["nodes"], dtype=float)
        density = np.asarray(frame["density"], dtype=float)
        label = frame["label"]
        center = coords.mean(axis=0)

        fig, ax = plt.subplots(figsize=self._figsize, dpi=150)
        ax.set_facecolor("0.92")
        fig.patch.set_facecolor("0.92")

        projected_domain, _ = self._camera_projection(coords, self._view_elev, self._view_azim, center=center)

        triangles = self._extract_matter_surface(coords, density, self._iso_value)
        if len(triangles) > 0:
            flat_points = triangles.reshape(-1, 3)
            projected_points, point_depth = self._camera_projection(flat_points, self._view_elev, self._view_azim, center=center)
            polygons = projected_points.reshape(-1, 3, 2)
            triangle_depth = point_depth.reshape(-1, 3).mean(axis=1)
            face_colors = self._binary_face_colors(triangles)
            order = np.argsort(triangle_depth)
            collection = PolyCollection(polygons[order], facecolors=face_colors[order], edgecolors=(0.05, 0.05, 0.05, 0.45), linewidths=0.18, antialiased=True)
            ax.add_collection(collection)

        self._draw_domain_edges(ax, coords, center)

        xmin, ymin = projected_domain.min(axis=0)
        xmax, ymax = projected_domain.max(axis=0)
        span = max(xmax - xmin, ymax - ymin, 1e-12)
        padding = 0.08 * span
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(label, fontsize=9, fontfamily="monospace")
        fig.tight_layout(pad=0.2)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, facecolor=fig.get_facecolor(), transparent=False)
        plt.close(fig)
        buffer.seek(0)
        image = np.asarray(Image.open(buffer).convert("RGB"))
        buffer.close()
        return image

    def save_mp4(self, path, fps=10):
        """Render all recorded frames and encode them as H.264 MP4."""
        if not self._frames:
            print("[DensityView] No frames recorded; MP4 not saved.")
            return

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        print(f"[DensityView] Rendering {len(self._frames)} frames...")
        arrays = [self._render_frame_rgb(frame) for frame in self._frames]

        height, width = arrays[0].shape[:2]
        padded_height = int(np.ceil(height / 16.0)) * 16
        padded_width = int(np.ceil(width / 16.0)) * 16

        if padded_height != height or padded_width != width:
            padded_arrays = []
            for array in arrays:
                padded = np.full((padded_height, padded_width, 3), 255, dtype=np.uint8)
                padded[:array.shape[0], :array.shape[1]] = array
                padded_arrays.append(padded)
            arrays = padded_arrays

        writer = imageio.get_writer(path, format="ffmpeg", fps=fps, codec="libx264", output_params=["-crf", "18"])
        try:
            for array in arrays:
                writer.append_data(array)
        finally:
            writer.close()

        print(f"[DensityView] MP4 saved to {path!r} ({len(arrays)} frames at {fps} fps).")

    @staticmethod
    def _standalone_gmsh_command(gmsh_executable, mesh_path):
        """Build a robust command for binary or Python-script Gmsh launchers."""
        options = [mesh_path, "-setnumber", "Mesh.SurfaceFaces", "1", "-setnumber", "Mesh.SurfaceEdges", "0", "-setnumber", "Mesh.Lines", "1", "-setnumber", "Mesh.Nodes", "0", "-setnumber", "General.Trackball", "1", "-setnumber", "General.Axes", "0"]
        try:
            with open(gmsh_executable, "rb") as stream:
                first_line = stream.readline().decode("utf-8", errors="ignore")
        except OSError:
            first_line = ""
        if "env python" in first_line:
            return [sys.executable, gmsh_executable, *options]
        return [gmsh_executable, *options]

    def export_interactive_mesh(self, frame=None, iso_value=None, path=None):
        """Export the closed material surface and initial-domain edges to a Gmsh file."""
        if frame is None:
            if not self._frames:
                raise RuntimeError("No frame has been recorded.")
            frame = self._frames[-1]

        iso_value = self._iso_value if iso_value is None else float(iso_value)
        mesh_path = os.path.abspath(self._interactive_mesh_path if path is None else path)
        os.makedirs(os.path.dirname(mesh_path) or ".", exist_ok=True)

        coords = np.asarray(frame["nodes"], dtype=float)
        density = np.asarray(frame["density"], dtype=float)
        triangles = self._extract_matter_surface(coords, density, iso_value)

        if len(triangles) == 0:
            raise RuntimeError(f"No material region rho >= {iso_value} was found.")

        vertices, faces = self._merge_triangle_vertices(triangles)
        previous_model = gmsh.model.getCurrent()
        model_name = "TopologyOptimizationResult"

        if model_name in gmsh.model.list():
            gmsh.model.setCurrent(model_name)
            gmsh.model.remove()

        gmsh.model.add(model_name)
        surface_tag = gmsh.model.addDiscreteEntity(2)
        surface_node_tags = np.arange(1, len(vertices) + 1, dtype=np.int64)
        surface_element_tags = np.arange(1, len(faces) + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(2, surface_tag, surface_node_tags.tolist(), vertices.ravel().tolist())
        gmsh.model.mesh.addElementsByType(surface_tag, 2, surface_element_tags.tolist(), (faces + 1).ravel().tolist())
        gmsh.model.setColor([(2, surface_tag)], int(self._COLOR[0]*255), int(self._COLOR[1]*255), int(self._COLOR[2]*255), 255)

        if self._show_domain_edges:
            box_corners, box_edges = self._initial_box_edges(coords)
            edge_tag = gmsh.model.addDiscreteEntity(1)
            edge_node_tags = np.arange(len(vertices) + 1, len(vertices) + len(box_corners) + 1, dtype=np.int64)
            edge_element_tags = np.arange(len(faces) + 1, len(faces) + len(box_edges) + 1, dtype=np.int64)
            gmsh.model.mesh.addNodes(1, edge_tag, edge_node_tags.tolist(), box_corners.ravel().tolist())
            gmsh.model.mesh.addElementsByType(edge_tag, 1, edge_element_tags.tolist(), edge_node_tags[box_edges].ravel().tolist())
            gmsh.model.setColor([(1, edge_tag)], 25, 25, 25, 255)

        gmsh.write(mesh_path)

        if previous_model and previous_model in gmsh.model.list():
            gmsh.model.setCurrent(previous_model)

        return mesh_path

    def show_interactive(self, frame=None, iso_value=0.5, wait=True, path=None):
        """Export and open the material surface in the standalone Gmsh GUI."""
        gmsh_executable = shutil.which("gmsh")
        if gmsh_executable is None:
            print("[DensityView] The standalone 'gmsh' executable was not found.")
            return None

        try:
            mesh_path = self.export_interactive_mesh(frame=frame, iso_value=iso_value, path=path)
        except RuntimeError as error:
            print(f"[DensityView] {error}")
            return None

        command = self._standalone_gmsh_command(gmsh_executable, mesh_path)
        print(f"[DensityView] Opening interactive surface: {mesh_path}")

        try:
            if wait:
                subprocess.run(command, check=False)
            else:
                subprocess.Popen(command)
        except OSError as error:
            print(f"[DensityView] Could not launch Gmsh: {error}")
            print(f"[DensityView] Open this file manually: {mesh_path}")

        return mesh_path
