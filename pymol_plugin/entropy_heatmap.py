"""Entropy heatmap visualization for FlexAID∆S binding modes.

Computes spatial entropy density from pose ensembles and renders
it as a pseudoatom-based heatmap in PyMOL (Phase 3, deliverable 3.1).

Usage:
    PyMOL> flexaids_load_results /path/to/output
    PyMOL> flexaids_entropy_heatmap 1
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

try:
    from pymol import cmd
except ImportError as exc:
    raise ImportError("PyMOL not available") from exc

try:
    from flexaidds import BindingModeResult, DockingResult, load_results
except ImportError as exc:
    raise ImportError(
        "flexaidds Python package is required for entropy heatmap"
    ) from exc


def _read_pose_coords(
    pdb_path: str,
    ligand_only: bool = True,
) -> List[Tuple[float, float, float]]:
    """Extract heavy-atom coordinates from a PDB file.

    Args:
        pdb_path: Path to PDB file.
        ligand_only: If True, only read HETATM records (ligand atoms).
            This dramatically reduces coordinate count and improves
            heatmap performance.  Set False to include protein atoms.

    Returns a list of (x, y, z) tuples for non-hydrogen atoms.
    """
    coords = []
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if ligand_only:
                    if not line.startswith("HETATM"):
                        continue
                else:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                element = line[76:78].strip() if len(line) >= 78 else ""
                atom_name = line[12:16].strip()
                if element == "H" or (not element and atom_name.startswith("H")):
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
    except (OSError, ValueError):
        pass
    # Fall back to all atoms if no HETATM found
    if not coords and ligand_only:
        return _read_pose_coords(pdb_path, ligand_only=False)
    return coords


def _compute_grid_bounds(
    all_coords: List[Tuple[float, float, float]],
    padding: float = 3.0,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute bounding box around all coordinates with padding."""
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    zs = [c[2] for c in all_coords]
    return (
        (min(xs) - padding, min(ys) - padding, min(zs) - padding),
        (max(xs) + padding, max(ys) + padding, max(zs) + padding),
    )


def _compute_spatial_entropy(
    mode: BindingModeResult,
    grid_spacing: float = 2.0,
    sigma: float = 2.0,
) -> List[Tuple[float, float, float, float]]:
    """Compute spatial entropy density on a 3D grid from pose ensemble.

    Each grid point accumulates a Gaussian-weighted atom count from all
    poses.  The local density is then converted to a Shannon-style
    entropy measure:  S_i = -sum_j p_j ln(p_j)  where p_j is the
    fraction of total density contributed by pose j at grid point i.

    Args:
        mode: BindingModeResult with pose PDB paths.
        grid_spacing: Distance between grid points in Angstroms.
        sigma: Gaussian smoothing width in Angstroms.

    Returns:
        List of (x, y, z, entropy) tuples for grid points with non-zero
        entropy.
    """
    if not mode.poses:
        return []

    # Collect coordinates from all poses
    pose_coords_list = []
    for pose in mode.poses:
        coords = _read_pose_coords(str(pose.path))
        if coords:
            pose_coords_list.append(coords)

    if not pose_coords_list:
        return []

    # Flatten all coordinates to compute grid bounds
    all_coords = [c for pose_coords in pose_coords_list for c in pose_coords]
    if not all_coords:
        return []

    (x_min, y_min, z_min), (x_max, y_max, z_max) = _compute_grid_bounds(all_coords)

    nx = max(1, int((x_max - x_min) / grid_spacing) + 1)
    ny = max(1, int((y_max - y_min) / grid_spacing) + 1)
    nz = max(1, int((z_max - z_min) / grid_spacing) + 1)

    # Cap grid size to avoid excessive computation
    max_points = 30
    nx = min(nx, max_points)
    ny = min(ny, max_points)
    nz = min(nz, max_points)

    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    n_poses = len(pose_coords_list)

    # Distance cutoff: beyond 4*sigma the Gaussian contribution is < 0.02%
    r2_cutoff = (4.0 * sigma) ** 2

    results = []

    for ix in range(nx):
        gx = x_min + ix * grid_spacing
        for iy in range(ny):
            gy = y_min + iy * grid_spacing
            for iz in range(nz):
                gz = z_min + iz * grid_spacing

                # Per-pose density at this grid point
                pose_densities = []
                for pose_coords in pose_coords_list:
                    density = 0.0
                    for ax, ay, az in pose_coords:
                        dx = gx - ax
                        dy = gy - ay
                        dz = gz - az
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 < r2_cutoff:
                            density += math.exp(-r2 * inv_2sigma2)
                    pose_densities.append(density)

                total = sum(pose_densities)
                if total < 1e-12:
                    continue

                # Shannon entropy over pose contributions
                entropy = 0.0
                for d in pose_densities:
                    p = d / total
                    if p > 1e-15:
                        entropy -= p * math.log(p)

                # Normalise to [0, 1] range (max entropy = ln(n_poses))
                max_s = math.log(n_poses) if n_poses > 1 else 1.0
                entropy_norm = entropy / max_s if max_s > 0 else 0.0

                # Only include points with meaningful density
                if total > 0.1:
                    results.append((gx, gy, gz, entropy_norm))

    return results


def _entropy_color(entropy_norm: float) -> Tuple[float, float, float]:
    """Map normalised entropy [0,1] to a blue-white-red colormap.

    0.0 (low entropy / ordered) -> blue
    0.5 (medium)                -> white
    1.0 (high entropy / disordered) -> red
    """
    t = max(0.0, min(1.0, entropy_norm))
    if t < 0.5:
        s = t / 0.5
        return (s, s, 1.0)  # blue -> white
    else:
        s = (t - 0.5) / 0.5
        return (1.0, 1.0 - s, 1.0 - s)  # white -> red


def render_entropy_heatmap(
    mode_id: int,
    grid_spacing: float = 2.0,
    sigma: float = 2.0,
    sphere_scale: float = 0.4,
    transparency: float = 0.3,
) -> None:
    """Compute and render an entropy heatmap for a loaded binding mode.

    Requires results to be loaded first via ``flexaids_load_results``.

    Args:
        mode_id: Numeric binding-mode identifier.
        grid_spacing: Grid spacing in Angstroms (default 2.0).
        sigma: Gaussian smoothing width in Angstroms (default 2.0).
        sphere_scale: PyMOL sphere scale for heatmap points.
        transparency: Sphere transparency (0=opaque, 1=invisible).

    Example:
        PyMOL> flexaids_entropy_heatmap 1
        PyMOL> flexaids_entropy_heatmap 1, grid_spacing=1.5, sigma=1.5
    """
    from . import results_adapter

    mode = results_adapter._get_mode(int(mode_id))
    if mode is None:
        print("ERROR: No loaded result set or mode not found. "
              "Use 'flexaids_load_results' first.")
        return

    if not mode.poses:
        print(f"ERROR: Mode {mode_id} has no poses.")
        return

    grid_spacing = float(grid_spacing)
    sigma = float(sigma)

    print(f"Computing entropy heatmap for mode {mode_id} "
          f"(grid={grid_spacing}A, sigma={sigma}A)...")

    grid_points = _compute_spatial_entropy(mode, grid_spacing, sigma)

    if not grid_points:
        print(f"WARNING: No entropy density computed for mode {mode_id}.")
        return

    obj_name = f"entropy_map_mode{mode_id}"

    # Remove previous heatmap if it exists
    try:
        cmd.delete(obj_name)
    except Exception:
        pass

    for i, (x, y, z, s_norm) in enumerate(grid_points):
        r, g, b = _entropy_color(s_norm)
        color_name = f"_ent_m{mode_id}_{i}"
        cmd.set_color(color_name, [r, g, b])
        cmd.pseudoatom(
            obj_name,
            pos=[x, y, z],
            name=f"E{i}",
            b=s_norm,
        )

    cmd.show("spheres", obj_name)
    cmd.set("sphere_scale", sphere_scale, obj_name)
    cmd.set("sphere_transparency", transparency, obj_name)

    # Color each pseudoatom by its entropy value
    for i, (x, y, z, s_norm) in enumerate(grid_points):
        color_name = f"_ent_m{mode_id}_{i}"
        cmd.color(color_name, f"{obj_name} and name E{i}")

    n_high = sum(1 for _, _, _, s in grid_points if s > 0.5)
    print(
        f"Entropy heatmap rendered: {len(grid_points)} grid points "
        f"({n_high} high-entropy). Object: '{obj_name}'"
    )
    print("  Blue = ordered (low entropy), Red = disordered (high entropy)")
