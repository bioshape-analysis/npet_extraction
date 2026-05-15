# libnpet/backends/centerline.py
"""
Numerical primitives for extracting a single MOLE-style centerline + per-point
radius from a tunnel void mask on a regular voxel grid.

The pipeline is:

    void mask (0.5A grid, C0 frame)
      -> binary_opening to prune side-channels narrower than r_open
      -> largest connected component containing the PTC voxel
      -> Euclidean Distance Transform (EDT) on the pruned mask
      -> Dijkstra source (PTC) -> sink (geodesically farthest void voxel)
         with cost = step_length * (1 + alpha / EDT_voxels)
      -> back-traced path, resampled to uniform arc length with a spline
      -> per-point inscribed_radius_A (EDT) and cross_section_radius_A
         (min over rays in the perpendicular plane, on the ORIGINAL mask)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_grid_spec(path: Path) -> Dict:
    """Load grid_spec_level_1.json. Returns dict with origin, voxel_size_A,
    shape, and the C0 transform anchors."""
    obj = json.loads(Path(path).read_text())
    return {
        "origin": np.asarray(obj["origin"], dtype=np.float32),
        "voxel_size_A": float(obj["voxel_size_A"]),
        "shape": tuple(int(x) for x in obj["shape"]),
        "ptc_world": np.asarray(obj["transform"]["ptc"], dtype=np.float32),
        "constriction_world": np.asarray(obj["transform"]["constriction"], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Mask preprocessing
# ---------------------------------------------------------------------------


def _ball_structuring_element(radius_vox: float) -> np.ndarray:
    """Boolean 3D ball of the given radius in voxel units. Radius < 0.5 collapses
    to no-op (single voxel)."""
    r = int(np.ceil(radius_vox))
    if r <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    coords = np.arange(-r, r + 1)
    dx, dy, dz = np.meshgrid(coords, coords, coords, indexing="ij")
    return (dx * dx + dy * dy + dz * dz) <= (radius_vox * radius_vox)


def prune_side_channels(
    mask: np.ndarray,
    open_radius_A: float,
    voxel_size_A: float,
    seed_ijk: Tuple[int, int, int],
) -> np.ndarray:
    """Binary-open with a ball of radius open_radius_A, then keep only the
    connected component containing seed_ijk. Returns a new bool mask. If the
    seed voxel falls outside the opened mask, returns the connected component
    nearest to it."""
    mask = mask.astype(bool, copy=False)
    r_vox = max(0.0, float(open_radius_A) / float(voxel_size_A))

    if r_vox >= 0.5:
        se = _ball_structuring_element(r_vox)
        opened = ndimage.binary_opening(mask, structure=se)
    else:
        opened = mask.copy()

    if not opened.any():
        return opened

    # 26-connectivity for component labelling (matches the Dijkstra graph).
    labels, n_labels = ndimage.label(opened, structure=np.ones((3, 3, 3), dtype=bool))
    if n_labels == 0:
        return np.zeros_like(opened)

    seed = tuple(int(x) for x in seed_ijk)
    seed_label = labels[seed] if opened[seed] else 0

    if seed_label == 0:
        # PTC voxel got eroded away by the opening. Fall back to the component
        # whose voxels are nearest the seed in Euclidean voxel distance.
        comp_idx = np.argwhere(opened)
        dists = np.linalg.norm(comp_idx - np.asarray(seed, dtype=np.int64), axis=1)
        seed_label = int(labels[tuple(comp_idx[int(np.argmin(dists))])])

    return labels == seed_label


# ---------------------------------------------------------------------------
# Endpoint selection
# ---------------------------------------------------------------------------


def voxel_nearest_point_c0(
    mask: np.ndarray, origin_c0: np.ndarray, voxel_size: float, point_c0: np.ndarray
) -> Tuple[int, int, int]:
    """Return the (i,j,k) of the in-mask voxel whose C0 center is closest to
    point_c0."""
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        raise ValueError("mask is empty")
    centers = origin_c0[None, :] + idx.astype(np.float32) * float(voxel_size)
    d2 = ((centers - point_c0[None, :].astype(np.float32)) ** 2).sum(axis=1)
    return tuple(int(x) for x in idx[int(np.argmin(d2))])


def find_endpoint_in_z_slab(
    mask: np.ndarray,
    edt_vox: np.ndarray,
    origin_c0: np.ndarray,
    voxel_size: float,
    *,
    z_target_A: float,
    slab_thickness_A: float = 3.0,
    fallback: bool = True,
) -> Tuple[int, int, int]:
    """Pick the in-mask voxel near z=z_target_A (in C0) with the maximum EDT.
    This finds the "center of the local tunnel cross-section" at a chosen
    z-plane, which is the right endpoint for a tunnel that's roughly aligned
    with the C0 z-axis.

    If no in-mask voxel falls inside the slab, falls back to widening the slab
    until something is found (unless fallback=False, then raises).
    """
    idx_all = np.argwhere(mask)
    if idx_all.shape[0] == 0:
        raise ValueError("mask is empty")

    z_centers = origin_c0[2] + idx_all[:, 2].astype(np.float32) * float(voxel_size)
    half = 0.5 * float(slab_thickness_A)

    in_slab = np.abs(z_centers - float(z_target_A)) <= half
    if not in_slab.any():
        if not fallback:
            raise ValueError(
                f"No in-mask voxels within {slab_thickness_A:.1f}A of z={z_target_A:.1f}"
            )
        # Pick nearest-in-z voxel (single closest by z-distance) as fallback
        nearest = int(np.argmin(np.abs(z_centers - float(z_target_A))))
        return tuple(int(x) for x in idx_all[nearest])

    slab_idx = idx_all[in_slab]
    edt_vals = edt_vox[slab_idx[:, 0], slab_idx[:, 1], slab_idx[:, 2]]
    return tuple(int(x) for x in slab_idx[int(np.argmax(edt_vals))])


def mask_z_range_A(
    mask: np.ndarray, origin_c0: np.ndarray, voxel_size: float
) -> Tuple[float, float]:
    """Return (z_min_A, z_max_A) for in-mask voxels in C0."""
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        raise ValueError("mask is empty")
    z = origin_c0[2] + idx[:, 2].astype(np.float32) * float(voxel_size)
    return float(z.min()), float(z.max())


# ---------------------------------------------------------------------------
# Voxel graph + Dijkstra
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS_26 = np.array(
    [(di, dj, dk)
     for di in (-1, 0, 1)
     for dj in (-1, 0, 1)
     for dk in (-1, 0, 1)
     if not (di == 0 and dj == 0 and dk == 0)],
    dtype=np.int32,
)


def build_voxel_graph(
    mask: np.ndarray, edt_vox: np.ndarray, voxel_size_A: float, alpha: float
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Build a sparse CSR adjacency matrix over in-mask voxels with edge cost
    = step_length_A * (1 + alpha / max(EDT_target, 1.0)).

    Returns
    -------
    graph : (N, N) csr_matrix
    void_idx : (N, 3) int32 array of (i, j, k) for each graph node
    voxel_to_node : flattened-grid -> node id (or -1 if not in mask)
    """
    mask = mask.astype(bool, copy=False)
    shape = mask.shape
    void_idx = np.argwhere(mask).astype(np.int32)
    n = void_idx.shape[0]
    if n == 0:
        raise ValueError("mask has no in-mask voxels")

    flat_mask = np.ravel_multi_index(void_idx.T, shape)
    voxel_to_node = -np.ones(int(np.prod(shape)), dtype=np.int64)
    voxel_to_node[flat_mask] = np.arange(n, dtype=np.int64)

    step_lens = np.linalg.norm(_NEIGHBOR_OFFSETS_26.astype(np.float32), axis=1) * float(voxel_size_A)

    rows_list = []
    cols_list = []
    data_list = []

    for k, off in enumerate(_NEIGHBOR_OFFSETS_26):
        nb = void_idx + off  # (N, 3)
        valid = (
            (nb[:, 0] >= 0) & (nb[:, 0] < shape[0])
            & (nb[:, 1] >= 0) & (nb[:, 1] < shape[1])
            & (nb[:, 2] >= 0) & (nb[:, 2] < shape[2])
        )
        if not valid.any():
            continue
        nb_v = nb[valid]
        flat_nb = np.ravel_multi_index(nb_v.T, shape)
        nb_node = voxel_to_node[flat_nb]
        keep = nb_node >= 0
        if not keep.any():
            continue

        src_nodes = np.flatnonzero(valid)[keep]
        dst_nodes = nb_node[keep]
        edt_dst = edt_vox[nb_v[keep, 0], nb_v[keep, 1], nb_v[keep, 2]].astype(np.float32)
        weights = step_lens[k] * (1.0 + float(alpha) / np.maximum(edt_dst, 1.0))

        rows_list.append(src_nodes.astype(np.int64))
        cols_list.append(dst_nodes.astype(np.int64))
        data_list.append(weights.astype(np.float32))

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    return graph, void_idx, voxel_to_node


def shortest_path_in_mask(
    mask: np.ndarray,
    edt_vox: np.ndarray,
    voxel_size_A: float,
    source_ijk: Tuple[int, int, int],
    alpha: float,
    *,
    target_ijk: Tuple[int, int, int] | None = None,
) -> Tuple[np.ndarray, Tuple[int, int, int], float]:
    """Run a single-source Dijkstra from the void voxel at source_ijk on a graph
    of in-mask voxels.

    If `target_ijk` is given and reachable, back-trace from there. Otherwise
    fall back to the geodesically-farthest reachable voxel.

    Returns the ordered (M, 3) array of voxel indices on the chosen path, the
    sink voxel index, and the geodesic cost to the sink.

    Cost field: edge u->v has weight step_length(u,v)_A * (1 + alpha/EDT(v)_vox),
    which biases the path toward high-clearance (medial) voxels but doesn't
    reward detours into wide pockets (the step-length term dominates for
    straight paths).
    """
    graph, void_idx, voxel_to_node = build_voxel_graph(mask, edt_vox, voxel_size_A, alpha)

    flat_src = int(np.ravel_multi_index(np.asarray(source_ijk, dtype=np.int64), mask.shape))
    src_node = int(voxel_to_node[flat_src])
    if src_node < 0:
        raise ValueError(f"source voxel {source_ijk} is not in mask")

    dist, predecessors = dijkstra(
        graph, directed=False, indices=src_node, return_predecessors=True
    )

    finite = np.isfinite(dist)
    if not finite.any():
        raise ValueError("Dijkstra returned no reachable voxels")

    sink_node = -1
    if target_ijk is not None:
        flat_tgt = int(np.ravel_multi_index(np.asarray(target_ijk, dtype=np.int64), mask.shape))
        tgt_node = int(voxel_to_node[flat_tgt])
        if tgt_node >= 0 and finite[tgt_node]:
            sink_node = tgt_node

    if sink_node < 0:
        # Fall back to geodesic-farthest reachable voxel.
        sink_node = int(np.argmax(np.where(finite, dist, -np.inf)))
    sink_dist = float(dist[sink_node])

    # back-trace
    path_nodes = []
    cur = sink_node
    while cur != -9999 and cur >= 0:
        path_nodes.append(cur)
        if cur == src_node:
            break
        cur = int(predecessors[cur])
    path_nodes.reverse()
    path_ijk = void_idx[np.asarray(path_nodes, dtype=np.int64)]
    sink_ijk = tuple(int(x) for x in void_idx[sink_node])
    return path_ijk.astype(np.int32), sink_ijk, sink_dist


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def voxel_path_to_c0(path_ijk: np.ndarray, origin_c0: np.ndarray, voxel_size_A: float) -> np.ndarray:
    return origin_c0[None, :] + path_ijk.astype(np.float32) * float(voxel_size_A)


def resample_polyline_uniform(
    points_c0: np.ndarray, step_A: float, smoothing: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spline-fit and resample a 3D polyline to uniform arc length. Returns
    (points, arc_lengths, unit_tangents)."""
    pts = np.asarray(points_c0, dtype=np.float64)
    if pts.shape[0] < 2:
        raise ValueError("need at least 2 points to resample")

    diffs = np.diff(pts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])

    if total < 2.0 * step_A or pts.shape[0] < 4:
        tangents = np.zeros_like(pts)
        tangents[1:-1] = pts[2:] - pts[:-2]
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(norms, 1e-9)
        return pts.astype(np.float32), arc.astype(np.float32), tangents.astype(np.float32)

    u = arc / total
    k = min(3, pts.shape[0] - 1)
    s = float(smoothing) * pts.shape[0]
    tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], u=u, s=s, k=k)

    n_new = max(2, int(np.ceil(total / float(step_A))) + 1)
    u_new = np.linspace(0.0, 1.0, n_new)
    pts_new = np.array(splev(u_new, tck)).T

    tangents = np.array(splev(u_new, tck, der=1)).T
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-9)

    diffs_new = np.diff(pts_new, axis=0)
    seg_new = np.linalg.norm(diffs_new, axis=1)
    arc_new = np.concatenate([[0.0], np.cumsum(seg_new)])

    return pts_new.astype(np.float32), arc_new.astype(np.float32), tangents.astype(np.float32)


# ---------------------------------------------------------------------------
# Radius readouts
# ---------------------------------------------------------------------------


def inscribed_radius_at_points(
    points_c0: np.ndarray,
    edt_vox: np.ndarray,
    origin_c0: np.ndarray,
    voxel_size_A: float,
) -> np.ndarray:
    """Trilinearly interpolated inscribed-sphere radius (Angstroms) at each
    point. The EDT is in voxel units."""
    coords = (points_c0.astype(np.float64) - origin_c0[None, :].astype(np.float64)) / float(voxel_size_A)
    r_vox = map_coordinates(edt_vox.astype(np.float32), coords.T, order=1, mode="nearest", cval=0.0)
    return (r_vox * float(voxel_size_A)).astype(np.float32)


def cross_section_radius_at_points(
    points_c0: np.ndarray,
    tangents_c0: np.ndarray,
    mask: np.ndarray,
    origin_c0: np.ndarray,
    voxel_size_A: float,
    n_rays: int = 64,
    max_search_A: float = 30.0,
    step_subvoxel: float = 0.25,
) -> np.ndarray:
    """At each centerline point, compute the minimum distance to the mask
    boundary along rays cast in the plane perpendicular to the local tangent.

    The minimum-over-rays naturally reports the local tube cross-section radius:
    a side-channel in one direction can't increase the minimum because the
    opposite (main-tube wall) direction still caps it.

    Operates on the ORIGINAL (not opened) mask so the reported geometry is real.
    """
    mask_b = mask.astype(bool, copy=False)
    shape = np.asarray(mask_b.shape, dtype=np.int64)
    step_A = float(step_subvoxel) * float(voxel_size_A)
    max_steps = max(2, int(np.ceil(float(max_search_A) / step_A)))
    dists = (np.arange(1, max_steps + 1, dtype=np.float32) * step_A)  # (S,)
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_rays), endpoint=False, dtype=np.float32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    n_pts = points_c0.shape[0]
    out = np.zeros(n_pts, dtype=np.float32)

    for i in range(n_pts):
        p = points_c0[i].astype(np.float32)
        t = tangents_c0[i].astype(np.float32)
        tn = np.linalg.norm(t)
        if tn < 1e-9:
            out[i] = 0.0
            continue
        t = t / tn
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = np.cross(t, ref)
        u = u / max(np.linalg.norm(u), 1e-9)
        v = np.cross(t, u)

        dirs = cos_t[:, None] * u[None, :] + sin_t[:, None] * v[None, :]  # (R, 3)
        samples = p[None, None, :] + dirs[:, None, :] * dists[None, :, None]  # (R, S, 3)
        ijk_f = (samples - origin_c0[None, None, :].astype(np.float32)) / float(voxel_size_A)
        ijk = np.floor(ijk_f).astype(np.int32)

        in_bounds = (
            (ijk[..., 0] >= 0) & (ijk[..., 0] < shape[0])
            & (ijk[..., 1] >= 0) & (ijk[..., 1] < shape[1])
            & (ijk[..., 2] >= 0) & (ijk[..., 2] < shape[2])
        )
        ijk_c = np.clip(ijk, 0, shape - 1)
        in_mask = mask_b[ijk_c[..., 0], ijk_c[..., 1], ijk_c[..., 2]]
        inside = in_bounds & in_mask  # (R, S)

        first_out = np.argmax(~inside, axis=1)  # 0 if all True
        all_in = inside.all(axis=1)
        first_out = np.where(all_in, np.int32(max_steps), first_out)

        ray_radii = step_A * first_out.astype(np.float32)
        out[i] = float(ray_radii.min())

    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_centerline_csv(
    out_path: Path,
    *,
    arc_length_A: np.ndarray,
    points_world: np.ndarray,
    points_c0: np.ndarray,
    inscribed_radius_A: np.ndarray,
    cross_section_radius_A: np.ndarray,
) -> None:
    """Emit the MOLE-style probe CSV. Columns:
    index, arc_length_A, x_world, y_world, z_world, x_C0, y_C0, z_C0,
    inscribed_radius_A, cross_section_radius_A
    """
    n = arc_length_A.shape[0]
    out_path = Path(out_path)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "arc_length_A",
            "x_world", "y_world", "z_world",
            "x_C0", "y_C0", "z_C0",
            "inscribed_radius_A", "cross_section_radius_A",
        ])
        for i in range(n):
            w.writerow([
                i,
                f"{float(arc_length_A[i]):.4f}",
                f"{float(points_world[i, 0]):.4f}",
                f"{float(points_world[i, 1]):.4f}",
                f"{float(points_world[i, 2]):.4f}",
                f"{float(points_c0[i, 0]):.4f}",
                f"{float(points_c0[i, 1]):.4f}",
                f"{float(points_c0[i, 2]):.4f}",
                f"{float(inscribed_radius_A[i]):.4f}",
                f"{float(cross_section_radius_A[i]):.4f}",
            ])


def write_centerline_ply(
    out_path: Path,
    points_world: np.ndarray,
    inscribed_radius_A: np.ndarray,
    cross_section_radius_A: np.ndarray,
    *,
    n_sides: int = 16,
    min_radius_A: float = 0.25,
) -> None:
    """Emit a triangulated tube mesh that sweeps the centerline, with each
    cross-section sized by `inscribed_radius_A`. PLY's polyline support and
    per-vertex scalar preservation are unreliable across writers and viewers
    (PyVista's PLY round-trip drops both), so a plain triangulated tube is the
    robust portable representation. The per-point radius numbers live in the
    companion CSV.
    """
    import pyvista as pv

    pts = np.asarray(points_world, dtype=np.float32)
    n = pts.shape[0]
    if n < 2:
        raise ValueError(f"centerline has too few points to sweep a tube ({n})")

    r_in = np.maximum(np.asarray(inscribed_radius_A, dtype=np.float32), float(min_radius_A))

    poly = pv.PolyData(pts)
    poly.lines = np.concatenate(([n], np.arange(n, dtype=np.int64))).astype(np.int64)
    poly["inscribed_radius_A"] = r_in

    tube = poly.tube(
        scalars="inscribed_radius_A",
        radius=float(min_radius_A),
        n_sides=int(n_sides),
        absolute=True,
    )
    tube.save(str(out_path), binary=True)


# ---------------------------------------------------------------------------
# Concentric-rings PLY (the visualization the user actually wants:
# each ring is a thin torus oriented perpendicular to the local tangent,
# major radius = local tunnel radius. Geometry alone encodes radius, so the
# file is self-contained and renders in PyMOL/ChimeraX/Mol*/MeshLab/Blender.)
# ---------------------------------------------------------------------------


def _build_ring_torus(
    center: np.ndarray,
    tangent: np.ndarray,
    ring_radius_A: float,
    tube_radius_A: float,
    n_circle: int,
    n_tube: int,
):
    import pyvista as pv

    t = np.asarray(tangent, dtype=np.float64)
    tn = float(np.linalg.norm(t))
    if tn < 1e-9:
        return None
    t = t / tn
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u = u / float(np.linalg.norm(u))
    v = np.cross(t, u)

    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_circle), endpoint=False)
    circle = (
        np.asarray(center, dtype=np.float64)[None, :]
        + float(ring_radius_A) * (
            np.cos(thetas)[:, None] * u[None, :]
            + np.sin(thetas)[:, None] * v[None, :]
        )
    )
    poly = pv.PolyData(circle.astype(np.float32))
    # closed polyline: returns to vertex 0
    poly.lines = np.concatenate(
        [[int(n_circle) + 1], np.arange(int(n_circle), dtype=np.int64), [0]]
    ).astype(np.int64)
    return poly.tube(radius=float(tube_radius_A), n_sides=int(n_tube))


def write_centerline_rings_ply(
    out_path: Path,
    points_world: np.ndarray,
    tangents_world: np.ndarray,
    radii_A: np.ndarray,
    *,
    ring_spacing_A: float = 1.0,
    tube_radius_A: float = 0.15,
    n_circle: int = 24,
    n_tube: int = 4,
    min_radius_A: float = 0.3,
) -> None:
    """Write a PLY containing one thin torus per centerline sample (subsampled
    at `ring_spacing_A`). Each torus is oriented perpendicular to the local
    tangent with major radius equal to the local tunnel radius.

    The file is fully self-contained: geometry alone encodes (centerline
    position, local radius, local axis direction). No per-vertex scalars
    needed, so the PLY round-trips through every viewer (PyMOL/ChimeraX/Mol*/
    MeshLab/Blender/Open3D)."""
    import pyvista as pv

    pts = np.asarray(points_world, dtype=np.float64)
    tan = np.asarray(tangents_world, dtype=np.float64)
    rad = np.asarray(radii_A, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 centerline points to draw rings (got {n})")

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(arc[-1])

    target_arcs = np.arange(0.0, total + 1e-9, float(ring_spacing_A))
    if target_arcs[-1] < total - 0.5 * float(ring_spacing_A):
        target_arcs = np.concatenate([target_arcs, [total]])
    ring_idx = np.searchsorted(arc, target_arcs).clip(0, n - 1)

    meshes = []
    for i in ring_idx:
        r = max(float(min_radius_A), float(rad[i]))
        ring = _build_ring_torus(
            pts[i], tan[i], r,
            tube_radius_A=float(tube_radius_A),
            n_circle=int(n_circle),
            n_tube=int(n_tube),
        )
        if ring is not None and ring.n_points > 0:
            meshes.append(ring)

    if not meshes:
        raise ValueError("no rings produced")

    combined = meshes[0]
    for m in meshes[1:]:
        combined = combined.merge(m)
    combined.save(str(out_path), binary=True)


# ---------------------------------------------------------------------------
# Pseudo-PDB (each centerline point = one HETATM, inscribed radius in
# B-factor, cross-section radius in occupancy). Universally portable: Mol*,
# PyMOL, ChimeraX, RasMol, VMD all render this as a chain of spheres sized
# by B-factor.
# ---------------------------------------------------------------------------


def write_centerline_pdb(
    out_path: Path,
    points_world: np.ndarray,
    inscribed_radius_A: np.ndarray,
    cross_section_radius_A: np.ndarray,
    *,
    connect_atoms: bool = True,
) -> None:
    """Write a pseudo-PDB where each centerline point is one CA atom of a
    fake residue 'TUN'. B-factor column = inscribed_radius_A (the MOLE-probe
    radius). Occupancy column = cross_section_radius_A. CONECT records draw
    bonds between consecutive atoms so the centerline renders as a line if
    `show sticks` is used.

    Load in Mol*: drag-and-drop into molstar.org/viewer; set representation
    to Spacefill; size = B-factor. Same idea in PyMOL/ChimeraX."""
    pts = np.asarray(points_world, dtype=np.float64)
    r_in = np.asarray(inscribed_radius_A, dtype=np.float64)
    r_cs = np.asarray(cross_section_radius_A, dtype=np.float64)
    n = pts.shape[0]

    lines = [
        "REMARK   1 NPET2 centerline pseudo-atoms",
        "REMARK   1 Each HETATM = one centerline point",
        "REMARK   1 B-factor column = inscribed_radius_A (MOLE probe semantics)",
        "REMARK   1 Occupancy column = cross_section_radius_A",
        "REMARK   1 Render as spheres sized by B-factor for the inscribed-",
        "REMARK   1 sphere envelope.",
    ]
    for i in range(n):
        serial = i + 1
        resi = (i % 9999) + 1
        chain = chr(ord("A") + min(25, i // 9999))
        occ = min(99.99, max(0.0, float(r_cs[i])))
        b = min(99.99, max(0.0, float(r_in[i])))
        # PDB cols (1-indexed):
        #   1-6   record name ("HETATM")
        #   7-11  atom serial (right-justified)
        #   13-16 atom name ("CA  ")
        #   18-20 residue name
        #   22    chain id
        #   23-26 residue seq number
        #   31-38 x
        #   39-46 y
        #   47-54 z
        #   55-60 occupancy
        #   61-66 b-factor
        #   77-78 element
        lines.append(
            f"HETATM{serial:>5d}  CA  TUN {chain}{resi:>4d}    "
            f"{pts[i,0]:8.3f}{pts[i,1]:8.3f}{pts[i,2]:8.3f}"
            f"{occ:6.2f}{b:6.2f}           C  "
        )

    if connect_atoms:
        for i in range(n - 1):
            lines.append(f"CONECT{i+1:>5d}{i+2:>5d}")

    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Viewer scripts (PyMOL .pml, ChimeraX .cxc). Both load: surface mesh +
# centerline rings + centerline pseudo-atoms, with sensible default styling.
# Paths are relative to the script's own directory.
# ---------------------------------------------------------------------------


def write_pymol_view_script(
    out_path: Path,
    *,
    surface_mesh_rel: str | None,
    rings_rel: str,
    atoms_rel: str,
) -> None:
    lines = [
        "# NPET2 centerline visualization for PyMOL.",
        "# Usage: `pymol view_pymol.pml` from this directory, or",
        "#        `@view_pymol.pml` inside PyMOL.",
        "bg_color white",
    ]
    if surface_mesh_rel:
        lines += [
            f"load {surface_mesh_rel}, tunnel_surface",
            "set transparency, 0.6, tunnel_surface",
            "color grey80, tunnel_surface",
        ]
    lines += [
        f"load {rings_rel}, centerline_rings",
        "color cyan, centerline_rings",
        "",
        f"load {atoms_rel}, centerline_atoms",
        "hide everything, centerline_atoms",
        "show spheres, centerline_atoms",
        "alter centerline_atoms, vdw=b",
        "rebuild",
        "spectrum b, blue_white_red, centerline_atoms",
        "",
        "orient",
        "zoom centerline_rings, 5",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def write_chimerax_view_script(
    out_path: Path,
    *,
    surface_mesh_rel: str | None,
    rings_rel: str,
    atoms_rel: str,
) -> None:
    lines = [
        "# NPET2 centerline visualization for ChimeraX.",
        "# Usage: `open view_chimerax.cxc` inside ChimeraX.",
        "set bgColor white",
    ]
    model_idx = 1
    if surface_mesh_rel:
        lines += [
            f"open {surface_mesh_rel}",
            f"transparency #{model_idx} 60 target s",
            f"color #{model_idx} light gray",
        ]
        model_idx += 1
    rings_idx = model_idx
    lines += [
        f"open {rings_rel}",
        f"color #{rings_idx} dodger blue",
    ]
    model_idx += 1
    atoms_idx = model_idx
    lines += [
        f"open {atoms_rel}",
        f"style #{atoms_idx} sphere",
        f"size #{atoms_idx} atomRadius byattribute bfactor",
        f"color bfactor #{atoms_idx} palette blue-white-red",
        "view",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Static overview PNG (pyvista offscreen). Compose: tunnel surface (translucent
# grey) + centerline rings (blue). Useful as a thumbnail in run dirs / reports.
# ---------------------------------------------------------------------------


def render_centerline_overview_png(
    out_path: Path,
    *,
    rings_path: Path,
    surface_mesh_path: Path | None = None,
    size: Tuple[int, int] = (1280, 960),
) -> None:
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")

    if surface_mesh_path is not None and Path(surface_mesh_path).exists():
        try:
            surf = pv.read(str(surface_mesh_path))
            pl.add_mesh(surf, color="lightgray", opacity=0.25, smooth_shading=True)
        except Exception:
            pass  # don't fail the PNG over a bad surface mesh

    rings = pv.read(str(rings_path))
    pl.add_mesh(rings, color="dodgerblue", smooth_shading=True)
    pl.add_axes()
    pl.screenshot(str(out_path), transparent_background=False)
    pl.close()
