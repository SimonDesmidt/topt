import os
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
os.environ["QT_QPA_PLATFORM"] = "xcb"

import gmsh
import shutil
import argparse
import subprocess
import numpy as np
from pathlib import Path
from ui_module import DensityView


def read_metadata(filename):
    metadata = {}

    with filename.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line: continue
            parts = line.split(maxsplit=1)

            if len(parts) != 2: raise ValueError(f"Invalid metadata line {line_number}: {line!r}")

            key, value = parts
            if key in {"num_nodes", "num_elements", "num_dofs", "iterations", "history_length"}:
                metadata[key] = int(value)
            else:
                metadata[key] = float(value)

    required = {"num_nodes", "num_elements", "num_dofs", "iterations", "history_length", "compliance"}
    missing = required.difference(metadata)

    if missing: raise ValueError(f"Missing metadata entries: {sorted(missing)}")
    return metadata


def read_exact_binary(filename, dtype, count):
    array = np.fromfile(filename, dtype=dtype)
    if array.size != count: raise ValueError(f"{filename} contains {array.size} values, but {count} were expected.")
    return array

def load_c_results(output_directory):
    metadata = read_metadata(output_directory / "metadata.txt")

    num_nodes = int(metadata["num_nodes"])
    num_elements = int(metadata["num_elements"])
    num_dofs = int(metadata["num_dofs"])
    history_length = int(metadata["history_length"])

    nodes = read_exact_binary(output_directory / "nodes.bin", np.dtype(np.float64), 3 * num_nodes).reshape(num_nodes, 3)
    elements = read_exact_binary(output_directory / "elements.bin", np.dtype(np.int32), 4 * num_elements).reshape(num_elements, 4)
    density = read_exact_binary(output_directory / "density.bin", np.dtype(np.float64), num_nodes)
    u = read_exact_binary(output_directory / "u.bin", np.dtype(np.float64), num_dofs)
    rhs = read_exact_binary(output_directory / "rhs.bin", np.dtype(np.float64), num_dofs)

    if history_length > 0:
        compliances = read_exact_binary(output_directory / "compliances.bin", np.dtype(np.float64), history_length)
        volume_fractions = read_exact_binary(output_directory / "volume_fractions.bin", np.dtype(np.float64), history_length)
        variations = read_exact_binary(output_directory / "variations.bin", np.dtype(np.float64), history_length)
        density_history = read_exact_binary(output_directory / "density_history.bin", np.dtype(np.float64), history_length * num_nodes).reshape(history_length, num_nodes)

    else:
        compliances = np.empty(0, dtype=np.float64)
        volume_fractions = np.empty(0, dtype=np.float64)
        variations = np.empty(0, dtype=np.float64)
        density_history = np.empty((0, num_nodes), dtype=np.float64)

    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]

    rhs_x = rhs[0::3]
    rhs_y = rhs[1::3]
    rhs_z = rhs[2::3]

    return {"nodes": nodes, "elements": elements, "density": density,
        "u": u, "ux": ux, "uy": uy, "uz": uz, "rhs": rhs, "rhs_x": rhs_x, "rhs_y": rhs_y, "rhs_z": rhs_z,
        "compliances": compliances, "volume_fractions": volume_fractions, "variations": variations,
        "density_history": density_history, "iterations": int(metadata["iterations"]),
        "history_length": history_length, "compliance": float(metadata["compliance"]), 
        "num_nodes": num_nodes, "num_elements": num_elements, "num_dofs": num_dofs}


def run_c_program(executable, output_directory, volfrac, meshsize, clean_output=True):
    executable = executable.resolve()
    output_directory = output_directory.resolve()

    if not executable.is_file(): raise FileNotFoundError(f"C executable not found: {executable}")
    if clean_output and output_directory.exists(): shutil.rmtree(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)
    command = [str(executable), str(output_directory), f"{volfrac:.17g}", f"{meshsize:.17g}"]

    print("Running:", " ".join(command))

    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.stdout: print(completed.stdout, end="")
    if completed.stderr: print(completed.stderr, end="")
    if completed.returncode != 0: raise RuntimeError(f"The C executable failed with return code {completed.returncode}.")
    return completed


def create_gmsh_model(nodes, elements, model_name="CDensityOptimization"):
    """
    Recreate the C-generated tetrahedral mesh as a discrete Gmsh model.

    Returns
    -------
    node_tags
        Gmsh node tags aligned with the rows of ``nodes``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    if nodes.ndim != 2 or nodes.shape[1] != 3: raise ValueError("nodes must have shape (num_nodes, 3).")
    if elements.ndim != 2 or elements.shape[1] != 4: raise ValueError("elements must have shape (num_elements, 4).")
    if np.any(elements < 0) or np.any(elements >= len(nodes)): raise ValueError("elements contains an invalid zero-based node index.")

    if not gmsh.isInitialized(): gmsh.initialize()

    if model_name in gmsh.model.list():
        gmsh.model.setCurrent(model_name)
        gmsh.model.remove()

    gmsh.model.add(model_name)

    volume_tag = gmsh.model.addDiscreteEntity(3)
    node_tags = np.arange(1, len(nodes) + 1, dtype=np.int64)
    element_tags = np.arange(1, len(elements) + 1, dtype=np.int64)

    gmsh.model.mesh.addNodes(3, volume_tag, node_tags.tolist(), nodes.ravel().tolist())

    # Gmsh element type 4 is a four-node linear tetrahedron.
    gmsh.model.mesh.addElementsByType(volume_tag, 4, element_tags.tolist(), (elements + 1).ravel().tolist())

    return node_tags


def populate_density_view(view, nodes, density_history, compliances):
    """
    Add all recorded C optimization iterations to a DensityView.
    """
    density_history = np.asarray(density_history, dtype=np.float64)
    compliances = np.asarray(compliances, dtype=np.float64)

    if density_history.ndim != 2:                    raise ValueError("density_history must have shape (iterations, num_nodes).")
    if density_history.shape[0] != len(compliances): raise ValueError("The density-history and compliance-history lengths do not match.")
    if density_history.shape[1] != len(nodes):       raise ValueError("The density-history node count does not match the mesh node count.")

    for iteration, (density, compliance) in enumerate(zip(density_history, compliances), start=1):
        view.update(nodes=nodes, density=density, iteration=iteration, compliance=float(compliance))


def main():
    parser = argparse.ArgumentParser()
    raw_output = Path("raw")
    npz_output = Path("outputs/results.npz")

    parser.add_argument("--exe", type=Path, default=Path("./topt"), help="Path to the compiled C executable.")
    parser.add_argument("--mp4", type=Path, default=Path("outputs/topt.mp4"), help="Output MP4 path.")
    parser.add_argument("--no-mp4", action="store_true", help="Do not render an MP4.")
    parser.add_argument("--interactive-mesh", type=Path, default=Path("outputs/c_topology_result.msh"), help="Path used for the final interactive Gmsh material surface.")
    parser.add_argument("--fps", type=int, default=10, help="MP4 frames per second.")
    parser.add_argument("--show-interactive", action="store_true", help="Open the final material surface in standalone Gmsh.")
    parser.add_argument("--show-gmsh-history", action="store_true", help="Open the Gmsh NodeData history after loading all iterations.")
    parser.add_argument("--skip-run", action="store_true", help="Load an existing C output directory without rerunning C.")
    parser.add_argument("--volfrac", type=float, default=0.10, help="Target material volume fraction.")
    parser.add_argument("--meshsize", type=float, default=0.10, help="Refinement of the mesh.")

    args = parser.parse_args()
    if not args.skip_run:
        run_c_program(executable=args.exe, output_directory=raw_output, volfrac=args.volfrac, meshsize=args.meshsize)

    results = load_c_results(raw_output)
    npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_output, **results)

    nodes = np.asarray(results["nodes"], dtype=np.float64)
    elements = np.asarray(results["elements"], dtype=np.int64)
    density = np.asarray(results["density"], dtype=np.float64)
    density_history = np.asarray(results["density_history"], dtype=np.float64)
    compliances = np.asarray(results["compliances"], dtype=np.float64)
    volume_fractions = np.asarray(results["volume_fractions"], dtype=np.float64)
    variations = np.asarray(results["variations"], dtype=np.float64)

    if len(density_history) == 0:
        density_history = density[None, :]
        compliances = np.array([float(results["compliance"])], dtype=np.float64)

    node_tags = create_gmsh_model(nodes, elements)

    view = DensityView(node_tags=node_tags, elements=elements, interactive_mesh_path=str(args.interactive_mesh))

    populate_density_view(view=view, nodes=nodes, density_history=density_history, compliances=compliances)
    
    print("\n======== Rendering ========")
    print(f"\nElements:               {len(elements)}\nNodes:                  {len(nodes)}")
    print(f"Number of iterations:   {len(density_history)}")
    print(f"Final compliance:       {compliances[-1]:.3e}")

    if len(volume_fractions) > 0: print(f"Final volume fraction:  {volume_fractions[-1]:.3f}")

    if not args.no_mp4:
        fps = max(1, int(args.fps))
        view.save_mp4(path=str(args.mp4), fps=fps)

    if args.show_interactive:
        view.show_interactive(frame=view._frames[-1], wait=True, path=str(args.interactive_mesh))

    if args.show_gmsh_history and gmsh.fltk.isAvailable():
        gmsh.fltk.run()

    if gmsh.isInitialized():
        gmsh.finalize()


if __name__ == "__main__":
    main()