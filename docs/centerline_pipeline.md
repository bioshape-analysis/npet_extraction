# The npet2 centerline pipeline

## Purpose

The exit-tunnel mesh that npet2 produces (Stage55GridRefine) is a watertight surface, which is what most downstream geometric analysis needs. But a large fraction of biology users want a single 1-D parameterization of the tunnel — coordinates along the axis plus a local radius at each station — to plot a radius profile, place residues in a "depth into tunnel" frame, or compare structures by their tunnel narrowing. That single-CSV summary was MOLE2's defining output for many years, and it is what Stage60Centerline reproduces.

This document explains how Stage60 computes that output, then compares the approach to MOLE2's probe method, calling out where the two are equivalent in spirit and where the discretization differences matter.

## Inputs

Stage60 reads two artifacts from Stage55GridRefine and two scalars from Stage10Landmarks:

- `selected_void_component_mask_level_1.npy` — a binary 3-D mask on a regular voxel grid (default voxel pitch 0.5 Å) covering the selected tunnel void in the C0 frame. The C0 frame is constructed in Stage10 such that the peptidyl transferase center sits at the origin and the constriction site lies on the positive z-axis, so to a first approximation the tunnel runs along +z.
- `grid_spec_level_1.json` — origin, voxel size, and shape of that grid, plus the world-frame coordinates of PTC and constriction needed to invert the C0 transform.
- `ptc_xyz`, `constriction_xyz` — the same world-frame landmarks Stage10 produced, so that final outputs can be expressed in both frames.

Stage60 does not reread the mesh or the atom coordinates. The mask is the only geometric input the centerline derives from. This is a design choice: by extracting from the same mask that the mesh was produced from, the centerline and the surface are guaranteed to be mutually consistent (the centerline always sits inside the rendered tube).

## Algorithm

The pipeline runs in seven steps.

**Step 1: prune side-channels by morphological opening.** Real tunnel voids that come out of Stage55 routinely include thin pockets and dead-end side-channels branching off the main tube. These cause two failure modes downstream: the shortest-path algorithm can detour into a wide pocket because its medial-axis cost is locally low, and the cross-section radius computed on the raw mask sees walls in the "wrong" places. Stage60 first runs a 3-D binary opening (`scipy.ndimage.binary_opening` with a ball structuring element of radius `centerline_open_radius_A`, default 1 Å) on the mask. Opening removes any protrusion narrower than the structuring element while leaving the main body of the mask largely intact. A side-channel narrower than the opening radius disappears entirely; one wider than the opening radius is unaffected but typically becomes disconnected because the opening also erodes around the junction.

**Step 2: keep only the connected component containing PTC.** After opening, the mask may have multiple connected components — the main tube plus stray pockets the opening didn't fully erase. Stage60 finds the in-mask voxel nearest the PTC in C0 (a rough "seed") and keeps only the 26-connected component containing it. This is the "centerline mask" used for everything that follows.

**Step 3: Euclidean distance transform.** `scipy.ndimage.distance_transform_edt` on the centerline mask returns, at each in-mask voxel, the Euclidean distance (in voxel units) to the nearest out-of-mask voxel. Multiplied by the voxel pitch, this is the radius of the largest sphere centered at that voxel that fits entirely inside the centerline mask — i.e. the local inscribed radius. This array is the cost field for the next step and the source of the per-point inscribed_radius_A in the CSV.

**Step 4: pick endpoints in z-slabs.** The C0 frame is constructed so the tunnel axis is approximately +z, which means the two ends of the tunnel are at z_min and z_max of the centerline mask. Stage60 takes a thin slab (default 3 Å thick) around each of those z-extremes and selects the in-mask voxel with the highest EDT within the slab as the endpoint. This is the centroid-of-cross-section heuristic: the highest-EDT voxel in a thin axial slab is the voxel closest to the "center" of the local tunnel cross-section. This step solves two real-data problems we saw on 5NWY without it: (a) when PTC sits outside the grid bounds, the "nearest in-mask voxel to PTC" lands on the wall and gives EDT ≈ 0, which makes the path start hugging the surface; (b) when the sink is chosen as the geodesically-farthest voxel from the source, it routinely lands on a side-cavity rather than the true exit, because side-cavities can be geodesically more distant than the actual exit despite being shorter on-axis.

**Step 5: Dijkstra on the voxel graph.** Nodes = in-mask voxels of the centerline mask. Edges = 26-connectivity to neighboring in-mask voxels. Edge cost from u to v is `step_length_A * (1 + alpha / EDT_voxels_at_v)`, where `alpha` is `centerline_path_alpha` (default 1.0) and `step_length_A` is the Euclidean voxel separation along that edge times the voxel pitch. The `1 + alpha/EDT` form is the small but important variation on the usual `1/EDT` Voronoi cost: pure `1/EDT` rewards detours into wide pockets (the cost drops faster than the path lengthens), while `1 + alpha/EDT` adds a baseline step-length component that prevents detours from winning. Dijkstra returns the shortest cost-weighted path from the source voxel; we back-trace from the sink. The graph is constructed in CSR form via `scipy.sparse.csgraph`.

**Step 6: spline resampling.** The raw path is a list of voxel-center positions at 0.5 Å pitch, which makes the curve visibly jagged because the discrete grid forces axis-aligned moves. Stage60 fits a 3-D cubic spline (`scipy.interpolate.splprep`) parameterized by arc length, then resamples at uniform `centerline_resample_step_A` (default 0.5 Å). The `centerline_smoothing` knob (default 0) controls how strictly the spline passes through the input voxel centers; nonzero values trade accuracy for visual smoothness. The spline also gives analytic tangents at each resampled point, which are needed for the cross-section radius.

**Step 7: per-point radii.** Two radii are reported.

The inscribed radius is trilinear interpolation of the EDT field (already in voxel units) times the voxel pitch. This is the MOLE2-probe semantic: at each centerline point, what is the radius of the maximum inscribed sphere centered there. It is computed on the *pruned* mask, so side-pockets that the opening already removed cannot contribute.

The cross-section radius is ray-marched in the plane perpendicular to the local centerline tangent on the *unpruned* mask. At each centerline point we cast `centerline_n_radial_rays` rays (default 64) radially outward in the perpendicular plane, march each ray in 0.125 Å steps, and record the distance at which it first leaves the mask. The reported cross-section radius is the minimum over rays. This is geometrically the radius of the largest disk in the local perpendicular plane that is fully contained in the void. Because it operates only in the perpendicular plane, off-axis pockets cannot inflate it — a side-channel branching perpendicular to the axis does not intersect the perpendicular plane at all, while a side-channel branching along the axis would have to extend through the main tube's wall to lie in the plane.

Both numbers are reported because they give different views of the same geometry. The inscribed radius is more familiar (it is MOLE2's probe radius), and is what most users expect to see in a probe CSV. The cross-section radius is more conservative and more directly answers "how wide is the tunnel here perpendicular to its axis". On a clean cylinder they agree to within one voxel; in a tunnel with side-cavities the inscribed radius can be larger than the cross-section radius wherever a cavity bulges out without disturbing the perpendicular cross-section.

## Outputs

Five files per run under `<run_dir>/stage/60_centerline/`:

- `centerline.csv` — the primary deliverable. Columns: `index, arc_length_A, x_world, y_world, z_world, x_C0, y_C0, z_C0, inscribed_radius_A, cross_section_radius_A`.
- `centerline.ply` — single ASCII PLY containing both a thin spine tube along the centerline path and rings (thin tori) perpendicular to the local tangent every 1 Å. Each ring's major radius equals the local inscribed_radius_A, so geometry alone encodes the radius profile; no per-vertex scalars are needed and the file is fully portable.
- `centerline_atoms.pdb` — pseudo-PDB with one HETATM per centerline point, inscribed radius in the B-factor column, cross-section radius in the occupancy column, and CONECT records between consecutive atoms. Mol*, PyMOL, ChimeraX, VMD all render this as a chain of spheres sized by B-factor.
- `view_pymol.pml` and `view_chimerax.cxc` — drop-in scripts loading the surface mesh + centerline PLY + pseudo-atoms with reasonable defaults.
- `centerline_overview.png` — pyvista-rendered thumbnail.
- `centerline_summary.json` — bottleneck location and value, path length, parameter values used, voxel counts.

## Configuration

The relevant fields in `RunConfig` (visible via `npet2 show-config`):

- `centerline_enabled` (default true) — set false to skip the stage.
- `centerline_open_radius_A` (default 1.0) — side-channel pruning aggressiveness. Larger values eliminate wider side-channels but also begin to erode true narrow sections like the constriction. 0.5 to 1.5 Å is the practical range.
- `centerline_path_alpha` (default 1.0) — clearance-vs-length tradeoff in the Dijkstra cost. Larger values push the path more strongly toward the medial axis; small values allow shorter paths near walls.
- `centerline_resample_step_A` (default 0.5) — arc-length spacing in the output CSV and along the spine.
- `centerline_smoothing` (default 0.0) — splprep smoothing factor. Zero forces the spline through every voxel-center; positive values smooth out grid jitter at the cost of accuracy at the endpoints.
- `centerline_n_radial_rays` (default 64) — number of rays per perpendicular plane in the cross-section radius computation.

A standalone CLI `npet2 centerline <RCSB_ID>` re-runs only this stage on an existing run, useful for sweeping `centerline_open_radius_A` or `centerline_path_alpha` without re-running the full pipeline.

## Comparison to MOLE2's probe method

MOLE2 is the most widely used predecessor for this kind of analysis, so it is worth being explicit about what is the same and what differs.

### What's the same in spirit

Both methods produce the same output type: an ordered sequence of points along the tunnel axis with a per-point radius. Both interpret the radius as the size of the largest sphere that fits at that point — the "probe radius" semantics. Both rely on a graph shortest-path through the tunnel interior to determine the path: MOLE2 uses Dijkstra on the Voronoi diagram of input atoms, npet2 uses Dijkstra on a regular voxel grid. Both use a cost function that rewards distance from the wall, so the path tends to hug the medial axis. Both require two endpoints — MOLE2 lets you specify them as residues or coordinates, npet2 derives them automatically from the C0 frame's z-extents.

### What's discretized differently

MOLE2 builds a 3-D Voronoi diagram of the input atom centers. Each Voronoi vertex is equidistant from its four nearest atoms, so its distance to the nearest atom (minus the atom radius) is the largest sphere that fits at that vertex without overlapping any atom. The path between two specified endpoints walks Voronoi edges, and the per-vertex sphere radii are the probe radii along it. The discretization is atom-density-dependent: the Voronoi has roughly one vertex per atom, and edges between adjacent vertices are the natural path candidates.

npet2 discretizes space on a regular voxel grid at 0.5 Å pitch instead. Each interior voxel's distance-to-wall comes from the Euclidean distance transform of the binary mask. The graph has one node per in-mask voxel and edges to 26-connected neighbors. The discretization is uniform in space rather than concentrated where atoms are.

The two approaches converge in the limit of fine discretization, but they have different failure modes at coarse discretization. MOLE2's Voronoi vertices are sparse where atoms are sparse, which can leave the medial axis poorly sampled in wide chambers. npet2's voxel grid is uniform, so wide chambers are sampled at the same fidelity as narrow ones, at the cost of allocating memory proportional to the bounding box rather than the atom count.

### What is genuinely different

The biggest methodological difference is what defines the "wall." MOLE2 treats each input atom as a sphere of a given radius (typically van der Waals), and the probe sphere may not overlap any atom-sphere. The Voronoi diagram is implicitly an inscribed-sphere-in-atomic-space representation. npet2 treats the wall as the surface of a binary occupancy mask whose construction (Stage55) already accounts for atomic radii, the cylindrical ROI cap, the alpha-shell envelope, and the morphological closing of voxel-discrete features. The wall is therefore "definition by mask" rather than "definition by atom-sphere union": more flexible (Stage55 can also include solvent, structural waters, etc.) but a step removed from the literal atomic structure.

A second difference is endpoint handling. MOLE2 requires the user to specify endpoints explicitly, usually as a known starting residue and a target on the protein surface. npet2 derives endpoints automatically from the z-extents of the centerline mask in the C0 frame, picking the highest-EDT voxel in a thin slab at each end. This is more automatic and reproducible across structures but assumes the tunnel runs along the C0 z-axis (an assumption that holds for ribosomal exit tunnels because Stage10 constructs C0 specifically that way, but would not generalize to arbitrary channels in other proteins).

A third difference is the second radius column. MOLE2 reports only the probe (inscribed) radius. npet2 additionally reports the cross-section radius — the minimum-over-rays in the local perpendicular plane. The cross-section radius is more conservative and more directly answers the question "is the tunnel locally cylindrical here, and if so, what is its cross-sectional radius?" The two columns differ noticeably wherever the local geometry is not locally cylindrical, which is informative on its own.

A fourth difference is branching. MOLE2 handles branched and multi-channel cavities natively by enumerating multiple paths through the Voronoi graph. npet2 in its current form picks a single source-to-sink path. For ribosomal exit tunnels this is the right model — the tunnel is overwhelmingly a single channel from PTC to the surface — but for cavities with genuine bifurcations a multi-path extension would be needed.

### Strengths of the npet2 approach

Tight pipeline integration. Stage60 reuses the void mask that Stage55 already produces, so the centerline and the rendered mesh are exact geometric companions of each other. There is no question of "which atomic radii were used for the centerline" — the centerline is defined by the same mask the mesh is.

Configurable side-channel pruning. The `centerline_open_radius_A` knob is an explicit lever the user can dial up or down to control how much off-axis cavity contributes to the result. The lever is intuitive (it is a length in Angstroms) and the effect on outputs is visible immediately. MOLE2's equivalent lever is the probe radius itself, which conflates "what we want to measure" with "what we want to ignore."

Cross-section radius as a separate, conservative measurement of "how wide is the main tunnel here." Useful as a sanity check against the inscribed radius and as a more defensible single number when reporting bottleneck widths.

Fine, isotropic resolution. The 0.5 Å voxel grid samples the medial axis at the same density everywhere in the tunnel, regardless of local atom density. For tunnels with a wide vestibule like the ribosomal exit, this gives smoother and more reliable radius profiles in the wide parts than a Voronoi-based method does.

Automatic endpoint selection that is reproducible across structures and does not require manual setup, as long as the C0 frame is constructed correctly upstream.

### Weaknesses of the npet2 approach

Voxel discretization. At 0.5 Å pitch the inscribed and cross-section radii are quantized at roughly that resolution, and the path itself can zig-zag by up to a voxel between adjacent positions. The spline resampling smooths this in the visualization but does not eliminate the underlying ~0.5 Å uncertainty in the radius profile. Halving the voxel pitch helps but costs 8× memory and ~8× compute on the EDT and Dijkstra steps.

Memory scales with bounding-box volume, not atom count. For the ribosomal exit tunnel ROI this is fine (~150 MB peak for the CSR graph at 0.5 Å), but a hypothetical larger or higher-resolution run could become expensive. MOLE2's Voronoi scales with atom count and is comparatively cheap.

Single-channel only in the current form. As noted, a genuine bifurcation would not be handled correctly — the algorithm would pick whichever branch the geodesic-farthest sink lands in. For ribosomal tunnels this is not in practice an issue, but it is a limitation worth knowing about.

Endpoint heuristic depends on the C0 frame's z-axis aligning with the tunnel. This is by construction true for npet2's ribosome-specific Stage10, but the centerline pipeline is not transferable to arbitrary cavity-extraction setups without revisiting the endpoint logic.

Mask quality is upstream. If Stage55 produces a mask with internal holes, disconnected fragments, or fused side-channels, those problems flow through to the centerline. MOLE2 derives its geometry directly from atomic coordinates each run and therefore does not inherit any prior segmentation artifacts.

## Practical use

The CSV is the source of truth for downstream analysis. The PLY and PDB are visual companions whose geometry is derived from (and is therefore consistent with) the CSV.

To plot a radius profile:

```python
import csv, matplotlib.pyplot as plt
rows = list(csv.DictReader(open("centerline.csv")))
arc  = [float(r["arc_length_A"]) for r in rows]
plt.plot(arc, [float(r["inscribed_radius_A"]) for r in rows], label="inscribed")
plt.plot(arc, [float(r["cross_section_radius_A"]) for r in rows], label="cross-section")
plt.xlabel("arc length from PTC (A)"); plt.ylabel("radius (A)"); plt.legend(); plt.show()
```

To compare two structures, align them in a common frame (npet2 already does this — both runs share the same C0 origin/orientation), then plot the two radius profiles against arc length on the same axes. Bottleneck positions and widths are directly comparable.
