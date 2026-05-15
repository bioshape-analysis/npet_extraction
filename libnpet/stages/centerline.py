# libnpet/stages/centerline.py
"""
Stage60Centerline: extract a MOLE-style probe centerline + radius CSV from the
selected void mask produced by Stage55GridRefine.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import ndimage

from libnpet.backends.centerline import (
    cross_section_radius_at_points,
    find_endpoint_in_z_slab,
    inscribed_radius_at_points,
    load_grid_spec,
    mask_z_range_A,
    prune_side_channels,
    render_centerline_overview_png,
    resample_polyline_uniform,
    shortest_path_in_mask,
    voxel_nearest_point_c0,
    voxel_path_to_c0,
    write_centerline_csv,
    write_centerline_pdb,
    write_centerline_ply,
    write_chimerax_view_script,
    write_pymol_view_script,
)
from libnpet.backends.geometry import get_transformation_to_C0, transform_points_from_C0
from libnpet.core.pipeline import Stage
from libnpet.core.types import ArtifactType, StageContext


class Stage60Centerline(Stage):
    key = "60_centerline"

    def params(self, ctx: StageContext) -> Dict[str, Any]:
        c = ctx.config
        return {
            "enabled"          : bool(getattr(c, "centerline_enabled", True)),
            "open_radius_A"    : float(getattr(c, "centerline_open_radius_A", 1.0)),
            "path_alpha"       : float(getattr(c, "centerline_path_alpha", 1.0)),
            "resample_step_A"  : float(getattr(c, "centerline_resample_step_A", 0.5)),
            "smoothing"        : float(getattr(c, "centerline_smoothing", 0.0)),
            "n_radial_rays"    : int(getattr(c, "centerline_n_radial_rays", 64)),
        }

    def run(self, ctx: StageContext) -> None:
        c = ctx.config
        if not bool(getattr(c, "centerline_enabled", True)):
            print(f"[{self.key}] disabled by config; skipping")
            return

        stage_dir = Path(ctx.store.stage_dir(self.key))
        stage_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------- inputs
        mask_path = Path(ctx.require("selected_void_component_mask_level_1_path"))
        spec_path = Path(ctx.require("grid_spec_level_1_path"))
        ptc_world = np.asarray(ctx.require("ptc_xyz"), dtype=np.float32)
        constr_world = np.asarray(ctx.require("constriction_xyz"), dtype=np.float32)

        mask_raw = np.load(mask_path).astype(bool)
        if not mask_raw.any():
            raise ValueError(f"[{self.key}] void mask at {mask_path} is empty")

        spec = load_grid_spec(spec_path)
        origin_c0 = spec["origin"].astype(np.float32)
        voxel_A = float(spec["voxel_size_A"])

        open_radius_A = float(getattr(c, "centerline_open_radius_A", 1.0))
        alpha = float(getattr(c, "centerline_path_alpha", 1.0))
        step_A = float(getattr(c, "centerline_resample_step_A", 0.5))
        smoothing = float(getattr(c, "centerline_smoothing", 0.0))
        n_rays = int(getattr(c, "centerline_n_radial_rays", 64))

        print(
            f"[{self.key}] mask={mask_raw.sum():,} vox  voxel={voxel_A}A  "
            f"open_r={open_radius_A}A  alpha={alpha}  step={step_A}A"
        )

        # ----------------------------------------------------- seed source roughly
        # The C0 frame has the tunnel axis ~aligned with +z, PTC at origin.
        # PTC itself can lie outside the grid bounds; pick the closest in-mask
        # voxel as a *seed* for the side-channel pruning step.
        ptc_c0 = np.zeros(3, dtype=np.float32)
        seed_ijk = voxel_nearest_point_c0(mask_raw, origin_c0, voxel_A, ptc_c0)

        # ------------------------------------------------------------- prune mask
        t0 = time.perf_counter()
        mask_pruned = prune_side_channels(
            mask_raw, open_radius_A=open_radius_A, voxel_size_A=voxel_A, seed_ijk=seed_ijk
        )
        n_pruned = int(mask_pruned.sum())
        if n_pruned == 0:
            raise ValueError(
                f"[{self.key}] pruning with open_radius_A={open_radius_A} left no voxels. "
                "Reduce centerline_open_radius_A."
            )
        print(f"[{self.key}] pruned mask: {n_pruned:,} vox  ({time.perf_counter()-t0:.2f}s)")

        # -------------------------------------------------------- EDT on pruned mask
        edt_vox = ndimage.distance_transform_edt(mask_pruned).astype(np.float32)

        # ----------------------------------------- principled source / sink picking
        # The mask's z-extents in C0 bracket the tunnel. Source = high-EDT voxel
        # in a thin slab at z_min; sink = high-EDT voxel in a thin slab at z_max.
        # This avoids two failure modes seen on real data:
        #   (a) PTC voxel sitting on the wall because PTC is at/outside the grid
        #       edge — would give EDT≈0 and a wall-hugging path start.
        #   (b) Geodesically-farthest sink veering into a side-cavity off the
        #       main +z tunnel axis — would terminate the path on a side branch.
        z_lo, z_hi = mask_z_range_A(mask_pruned, origin_c0, voxel_A)
        slab_thickness_A = max(3.0, 6.0 * voxel_A)
        src_ijk = find_endpoint_in_z_slab(
            mask_pruned, edt_vox, origin_c0, voxel_A,
            z_target_A=z_lo, slab_thickness_A=slab_thickness_A,
        )
        sink_seed_ijk = find_endpoint_in_z_slab(
            mask_pruned, edt_vox, origin_c0, voxel_A,
            z_target_A=z_hi, slab_thickness_A=slab_thickness_A,
        )
        print(
            f"[{self.key}] z-range=[{z_lo:.1f}, {z_hi:.1f}]A  "
            f"src={src_ijk} (EDT={edt_vox[src_ijk]*voxel_A:.2f}A)  "
            f"sink_seed={sink_seed_ijk} (EDT={edt_vox[sink_seed_ijk]*voxel_A:.2f}A)"
        )

        # ------------------------------------------------------------ Dijkstra
        # We run Dijkstra to the geodesic-farthest voxel, then trace the path
        # to sink_seed_ijk if it's reachable, else use the geodesic sink.
        t1 = time.perf_counter()
        path_ijk, geo_sink_ijk, geo_dist = shortest_path_in_mask(
            mask_pruned, edt_vox, voxel_A, source_ijk=src_ijk, alpha=alpha,
            target_ijk=sink_seed_ijk,
        )
        sink_ijk = tuple(int(x) for x in path_ijk[-1])
        print(
            f"[{self.key}] Dijkstra: {path_ijk.shape[0]:,} voxels on path, "
            f"sink at {sink_ijk}, geodesic cost={geo_dist:.2f}A  "
            f"({time.perf_counter()-t1:.2f}s)"
        )
        if path_ijk.shape[0] < 2:
            raise ValueError(
                f"[{self.key}] degenerate path with {path_ijk.shape[0]} voxels"
            )

        # -------------------------------------- resample to uniform arc length
        raw_path_c0 = voxel_path_to_c0(path_ijk, origin_c0, voxel_A)
        pts_c0, arc_A, tan_c0 = resample_polyline_uniform(
            raw_path_c0, step_A=step_A, smoothing=smoothing
        )
        total_len = float(arc_A[-1])
        print(f"[{self.key}] resampled to {pts_c0.shape[0]} points, total length {total_len:.2f}A")

        # ---------------------------------------------------- radii at each point
        inscribed_A = inscribed_radius_at_points(pts_c0, edt_vox, origin_c0, voxel_A)
        cross_A = cross_section_radius_at_points(
            pts_c0, tan_c0, mask_raw, origin_c0, voxel_A,
            n_rays=n_rays,
            max_search_A=float(getattr(c, "cylinder_radius_A", 35.0)) * 1.2,
        )

        # ------------------------------------------------ world-frame coordinates
        pts_world = transform_points_from_C0(pts_c0.astype(np.float32), ptc_world, constr_world).astype(np.float32)
        # Tangent vectors are directions, so only the rotation applies.
        _, rot = get_transformation_to_C0(ptc_world, constr_world)
        tan_world = (tan_c0.astype(np.float64) @ rot).astype(np.float32)

        # ---------------------------------------------------------- write outputs
        csv_path = stage_dir / "centerline.csv"
        write_centerline_csv(
            csv_path,
            arc_length_A=arc_A,
            points_world=pts_world,
            points_c0=pts_c0,
            inscribed_radius_A=inscribed_A,
            cross_section_radius_A=cross_A,
        )

        # Single PLY: thin spine tube along the path + rings perpendicular to
        # the tangent at every 1A, ring radius = local inscribed_radius_A.
        # ASCII format because some viewers (Mol*) require it.
        ply_path = stage_dir / "centerline.ply"
        write_centerline_ply(
            ply_path,
            pts_world, tan_world, inscribed_A,
            ring_spacing_A=max(1.0, 2.0 * step_A),
            ascii=True,
        )

        pdb_path = stage_dir / "centerline_atoms.pdb"
        write_centerline_pdb(pdb_path, pts_world, inscribed_A, cross_A)

        # Viewer scripts. Paths are relative to this stage dir so the scripts
        # work no matter where the run is moved. Prefer the ASCII surface mesh
        # (Stage55 emits both binary and ASCII side-by-side).
        s55_dir = Path(ctx.store.stage_dir("55_grid_refine"))
        ascii_surf = s55_dir / "mesh_level_1_ascii.ply"
        bin_surf = s55_dir / "mesh_level_1.ply"
        surface_mesh_path = ascii_surf if ascii_surf.exists() else bin_surf
        surface_mesh_rel = (
            f"../55_grid_refine/{surface_mesh_path.name}"
            if surface_mesh_path.exists() else None
        )
        pml_path = stage_dir / "view_pymol.pml"
        write_pymol_view_script(
            pml_path,
            surface_mesh_rel=surface_mesh_rel,
            centerline_ply_rel=ply_path.name,
            atoms_rel=pdb_path.name,
        )
        cxc_path = stage_dir / "view_chimerax.cxc"
        write_chimerax_view_script(
            cxc_path,
            surface_mesh_rel=surface_mesh_rel,
            centerline_ply_rel=ply_path.name,
            atoms_rel=pdb_path.name,
        )

        # Static overview PNG (best-effort; offscreen rendering can fail on
        # some headless boxes, don't take the whole stage down for it).
        png_path = stage_dir / "centerline_overview.png"
        try:
            render_centerline_overview_png(
                png_path,
                centerline_ply_path=ply_path,
                surface_mesh_path=surface_mesh_path if surface_mesh_path.exists() else None,
            )
        except Exception as e:
            print(f"[{self.key}] overview PNG render failed: {e}")
            png_path = None

        # ------------------------------------------------------- summary metrics
        bottleneck_idx = int(np.argmin(inscribed_A))
        summary = {
            "n_points"                : int(pts_c0.shape[0]),
            "total_length_A"          : float(total_len),
            "min_inscribed_radius_A"  : float(inscribed_A.min()),
            "min_cross_section_radius_A": float(cross_A.min()),
            "bottleneck_index"        : bottleneck_idx,
            "bottleneck_arc_length_A" : float(arc_A[bottleneck_idx]),
            "bottleneck_xyz_world"    : [float(x) for x in pts_world[bottleneck_idx].tolist()],
            "bottleneck_xyz_c0"       : [float(x) for x in pts_c0[bottleneck_idx].tolist()],
            "source_voxel_ijk"        : list(src_ijk),
            "sink_voxel_ijk"          : list(sink_ijk),
            "open_radius_A"           : float(open_radius_A),
            "path_alpha"              : float(alpha),
            "resample_step_A"         : float(step_A),
            "n_radial_rays"           : int(n_rays),
            "voxel_size_A"            : voxel_A,
            "pruned_mask_voxels"      : n_pruned,
            "raw_mask_voxels"         : int(mask_raw.sum()),
        }

        print(
            f"[{self.key}] bottleneck: inscribed={inscribed_A.min():.2f}A  "
            f"cross_section={cross_A.min():.2f}A  at arc={arc_A[bottleneck_idx]:.1f}A"
        )

        # ---------------------------------------------------- register artifacts
        ctx.store.register_file(
            name="centerline_csv",
            stage=self.key,
            type=ArtifactType.CSV,
            path=csv_path,
            meta={"columns": [
                "index", "arc_length_A",
                "x_world", "y_world", "z_world",
                "x_C0", "y_C0", "z_C0",
                "inscribed_radius_A", "cross_section_radius_A",
            ]},
            depends_on=("refined_surface_points_level_1",),
        )
        ctx.store.register_file(
            name="centerline_ply",
            stage=self.key,
            type=ArtifactType.PLY_MESH,
            path=ply_path,
            meta={
                "format": "ascii",
                "contents": "spine_tube + rings_perpendicular_to_tangent",
                "ring_radius_meaning": "inscribed_radius_A",
            },
            depends_on=("centerline_csv",),
        )
        ctx.store.register_file(
            name="centerline_atoms",
            stage=self.key,
            type=ArtifactType.PDB,
            path=pdb_path,
            meta={
                "b_factor_meaning": "inscribed_radius_A",
                "occupancy_meaning": "cross_section_radius_A",
            },
            depends_on=("centerline_csv",),
        )
        ctx.store.register_file(
            name="view_pymol",
            stage=self.key,
            type=ArtifactType.TXT,
            path=pml_path,
            meta={"viewer": "pymol"},
        )
        ctx.store.register_file(
            name="view_chimerax",
            stage=self.key,
            type=ArtifactType.TXT,
            path=cxc_path,
            meta={"viewer": "chimerax"},
        )
        if png_path is not None and png_path.exists():
            ctx.store.register_file(
                name="centerline_overview_png",
                stage=self.key,
                type=ArtifactType.PNG,
                path=png_path,
                depends_on=("centerline_ply",),
            )
        ctx.store.put_json(
            name="centerline_summary",
            stage=self.key,
            obj=summary,
            meta={"units": "A"},
        )

        # ------------------------------------------------ stash for downstream
        ctx.inputs["centerline_csv_path"] = str(csv_path)
        ctx.inputs["centerline_ply_path"] = str(ply_path)
        ctx.inputs["centerline_summary"] = summary
