import pyvista as pv
import numpy as np
import argparse
import itertools
import tqdm
import concurrent.futures
from pathlib import Path
import glob
import json
import os
import pandas as pd

from cr_nematic_structure.cr_nematic_structure import Configuration


def get_last_output_path():
    return Path(sorted(glob.glob("out/*"))[-1])


def get_all_iterations(output_path: Path | None = None):
    if output_path is None:
        output_path = get_last_output_path()
    folders = glob.glob(str(output_path / "cells/json") + "/*")
    iterations = [int(Path(fi).name) for fi in folders]
    return output_path, iterations


def load_cells_from_iteration(output_path: Path, iteration: int):
    load_path = Path(output_path) / "cells/json/{:020}".format(iteration)
    results = []
    for file in glob.glob(str(load_path) + "/*"):
        f = open(file)
        elements = json.load(f)["data"]
        elements = [element["element"][0] for element in elements]
        results.extend(elements)
    df = pd.json_normalize(results)

    if len(df) > 0:
        # Format individual entries for easier use later on
        # df["identifier"] = df["identifier"].apply(lambda x: tuple(x))
        df["cell.mechanics.pos"] = df["cell.mechanics.pos"].apply(
            lambda x: np.array(x[0], dtype=float).reshape(3, -1)
        )
        df["cell.mechanics.vel"] = df["cell.mechanics.vel"].apply(
            lambda x: np.array(x[0], dtype=float)
        )

    return df


def get_cell_meshes(iteration: int, path: Path, growth_rate_max):
    cells = load_cells_from_iteration(path, iteration)
    positions = [x for x in cells["cell.mechanics.pos"]]
    radii = np.array([x for x in cells["cell.interaction.radius"]], dtype=float)
    neighbors = np.array([x for x in cells["cell.neighbors"]], dtype=float)
    cell_surfaces = []
    for i, p in enumerate(positions):
        meshes = []
        # Add the sphere at the first point
        meshes.append(pv.Sphere(center=p[:, 0], radius=radii[i]))
        for j in range(max(p.shape[1] - 1, 0)):
            # Add a sphere for each following point
            meshes.append(pv.Sphere(center=p[:, j + 1], radius=radii[i]))

            # Otherwise add cylinders
            pos1 = p[:, j]
            pos2 = p[:, j + 1]
            center = 0.5 * (pos1 + pos2)
            direction = pos2 - center
            radius = radii[i]
            height = float(np.linalg.norm(pos1 - pos2))
            cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=height)
            meshes.append(cylinder)
        combined = pv.MultiBlock(meshes).combine()
        if combined is None:
            return None
        merged = combined.extract_surface()
        if merged is None:
            return None
        merged = merged.clean()
        merged["color"] = np.repeat(neighbors[i], len(merged.points))
        cell_surfaces.append(merged)
    return cell_surfaces


def plot_spheres(
    iteration: int,
    domain_size: tuple[float, float, float],
    growth_rate_max: float,
    neighbor_cap: int,
    path: Path = Path("./"),
    opath: Path | None = None,
    overwrite: bool = False,
    transparent_background: bool = False,
):
    if opath is None:
        opath = path / "images/{:010}.png".format(iteration)
        opath.parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(opath) and overwrite is False:
        return None
    cell_meshes = get_cell_meshes(iteration, path, growth_rate_max)
    cell_meshes = cell_meshes if cell_meshes is not None else []

    # General Settings
    plotter = pv.Plotter(off_screen=True, window_size=[1280, 1280])
    plotter.set_background([100, 100, 100])

    # Draw box around everything
    box = pv.Box(bounds=(0, domain_size[0], 0, domain_size[1], 0, domain_size[1]))
    plotter.add_mesh(box, style="wireframe", line_width=15)

    for cell in cell_meshes:
        plotter.add_mesh(
            cell,
            show_edges=False,
            scalars="color",
            clim=(0, neighbor_cap),
            cmap="summer",
            diffuse=0.5,
            ambient=0.5,
        )

    # Define camera
    dx = 0.05 * np.array(domain_size)
    lower = -dx
    upper = +dx + np.array(domain_size)
    bounds = (lower[0], upper[0], lower[1], upper[1], lower[2], upper[2])
    pv.Plotter.view_xy(plotter, bounds=bounds)
    pv.Plotter.enable_ssao(plotter, radius=12)
    plotter.enable_anti_aliasing()
    img = plotter.screenshot(opath, transparent_background=transparent_background)
    plotter.close()
    return img


def __plot_spheres_helper(args):
    (args, kwargs) = args
    plot_spheres(*args, **kwargs)


def plot_all_spheres(
    path: Path,
    config: Configuration,
    n_threads: int | None = None,
    overwrite: bool = False,
    transparent_background: bool = False,
):
    iterations = [it for it in get_all_iterations(path)[1]]
    if n_threads is None:
        n_cpu = os.cpu_count()
        if n_cpu is not None and n_cpu > 2:
            n_threads = n_cpu - 2
        else:
            n_threads = 1
    args = [(it,) for it in iterations]
    kwargs = {
        "domain_size": config.domain_size,
        "growth_rate_max": config.growth_rate,
        "neighbor_cap": config.neighbor_cap,
        "path": path,
        "overwrite": overwrite,
        "transparent_background": transparent_background,
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
        _ = list(
            tqdm.tqdm(
                executor.map(__plot_spheres_helper, zip(args, itertools.repeat(kwargs))),
                total=len(iterations),
            )
        )


# TODO actually use this
if __name__ == "_main__":
    parser = argparse.ArgumentParser(
        prog="Plotter",
        description="A plotting CLI for the springs example",
        epilog="For suggestions or bug-reports please refer to\
            https://github.com/jonaspleyer/cellular_raza/",
    )
    parser.add_argument(
        "-n",
        "--iteration",
        help="Specify which iteration to plot.\
            Multiple arguments are accepted as a list.\
            If not specified, plot all iterations.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input path of files. If not given,\
            it will be determined automatically by searching in './out/'",
    )
    parser.add_argument("-o", "--output-path", help="Path where to store generated images.")

    args = parser.parse_args()

if __name__ == "__main__":
    print("Plotting Individual Snapshots")
    output_path = get_last_output_path()
    plot_all_spheres(output_path, transparent_background=True, overwrite=True)
    print("Generating Movie")
    bashcmd = f"ffmpeg\
        -v quiet\
        -stats\
        -y\
        -r 30\
        -f image2\
        -pattern_type glob\
        -i '{output_path}/images/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -strict -2 {output_path}/movie.mp4"
    os.system(bashcmd)
    print("Playing Movie")
    bashcmd2 = f"firefox {output_path}/movie.mp4"
    os.system(bashcmd2)
