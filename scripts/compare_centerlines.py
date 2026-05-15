"""Compare exit-tunnel centerlines across many npet2 runs.

Walks ~/.npet2/runs/ (or whatever NPET2_RUNS_ROOT points at), finds the most
recent run per RCSB_ID, reads its stage/60_centerline/centerline.csv, and
produces:

  - a single-panel overlay of inscribed_radius_A vs arc_length_A for every
    structure found, color-coded by kingdom
  - a four-panel grid (one panel per kingdom: bacteria, archaea, eukarya,
    organellar) plotting the same data with shared axes
  - a CSV summary of bottleneck radius and tunnel length per structure

Usage:
    python scripts/compare_centerlines.py                     # all structures it can find
    python scripts/compare_centerlines.py 5AFI 6EK0 4V9F      # specific PDB IDs
    python scripts/compare_centerlines.py --runs-root /path   # alt runs root
    python scripts/compare_centerlines.py --radius cross      # use cross-section radius
    python scripts/compare_centerlines.py --out figs/         # write outputs to figs/

Outputs default to the current directory:
    centerline_overlay.png
    centerline_by_kingdom.png
    centerline_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Kingdom assignment for the paper-list structures ----------------------
# Extend this map as you add more PDBs.
KINGDOM_BY_PDB: Dict[str, str] = {
    # bacteria
    "5NJT": "bacteria", "5AFI": "bacteria", "5JVG": "bacteria",
    "5MYJ": "bacteria", "5O60": "bacteria", "5V7Q": "bacteria",
    "5NRG": "bacteria", "4Y4P": "bacteria", "3J7Z": "bacteria",
    "5VP2": "bacteria",
    # archaea
    "4V9F": "archaea", "4V6U": "archaea",
    # eukarya
    "6EK0": "eukarya", "5T2A": "eukarya", "3J79": "eukarya",
    "5GAK": "eukarya", "4V7E": "eukarya", "5T5H": "eukarya",
    "5XXB": "eukarya", "5XY3": "eukarya", "4UG0": "eukarya",
    # organellar
    "5X8T": "organellar", "3J9M": "organellar",
    # extras we already have
    "5NWY": "bacteria",  # E. coli Polikanov 2014
    "9RHU": "bacteria",  # E. coli stalled, with nascent chain G2
}

KINGDOM_COLORS = {
    "bacteria":   "tab:blue",
    "archaea":    "tab:green",
    "eukarya":    "tab:red",
    "organellar": "tab:purple",
}

KINGDOM_ORDER = ("bacteria", "archaea", "eukarya", "organellar")


def _runs_root(override: Optional[Path]) -> Path:
    if override is not None:
        return override
    env = os.environ.get("NPET2_RUNS_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".npet2" / "runs"


def _latest_run_dir(struct_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified run directory under struct_dir, or
    None if there are none."""
    if not struct_dir.is_dir():
        return None
    candidates = [d for d in struct_dir.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _load_centerline_csv(csv_path: Path) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {
        "arc_length_A": [],
        "inscribed_radius_A": [],
        "cross_section_radius_A": [],
    }
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in cols:
                cols[k].append(float(row[k]))
    return cols


def gather_centerlines(
    runs_root: Path, pdb_ids: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Return {pdb_id: {"arc_length_A": [...], "inscribed_radius_A": [...],
    "cross_section_radius_A": [...], "run_dir": Path}} for every PDB ID with
    an available centerline.csv."""
    out: Dict[str, Dict] = {}
    if pdb_ids:
        struct_ids = [p.upper() for p in pdb_ids]
    else:
        struct_ids = sorted(d.name for d in runs_root.iterdir() if d.is_dir())

    for sid in struct_ids:
        run = _latest_run_dir(runs_root / sid)
        if run is None:
            print(f"  skip {sid}: no run dir under {runs_root / sid}", file=sys.stderr)
            continue
        csv_path = run / "stage" / "60_centerline" / "centerline.csv"
        if not csv_path.exists():
            print(
                f"  skip {sid}: no centerline.csv (run was extracted before Stage60 — "
                f"`npet2 centerline {sid}` will add it without re-running the rest)",
                file=sys.stderr,
            )
            continue
        cols = _load_centerline_csv(csv_path)
        cols["run_dir"] = run
        out[sid] = cols
    return out


def write_summary_csv(
    data: Dict[str, Dict], out_path: Path, radius_field: str
) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pdb_id", "kingdom", "n_points", "total_length_A",
            f"min_{radius_field}", f"min_arc_length_A",
            f"mean_{radius_field}", f"median_{radius_field}",
            f"max_{radius_field}", "run_dir",
        ])
        for sid in sorted(data.keys()):
            d = data[sid]
            r = d[radius_field]
            arc = d["arc_length_A"]
            min_i = r.index(min(r))
            sorted_r = sorted(r)
            median = sorted_r[len(sorted_r) // 2]
            w.writerow([
                sid,
                KINGDOM_BY_PDB.get(sid, "?"),
                len(arc),
                f"{arc[-1]:.2f}",
                f"{min(r):.3f}",
                f"{arc[min_i]:.2f}",
                f"{sum(r)/len(r):.3f}",
                f"{median:.3f}",
                f"{max(r):.3f}",
                str(d["run_dir"]),
            ])


def _smooth_and_trim(
    arc: List[float],
    rad: List[float],
    smooth_sigma_A: float,
    trim_A: float,
    resample_step_A: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """Drop `trim_A` from each end of the curve, then Gaussian-smooth the
    radius with sigma=`smooth_sigma_A`."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    a = np.asarray(arc, dtype=float)
    r = np.asarray(rad, dtype=float)
    if a.size < 5:
        return list(a), list(r)

    keep = (a >= trim_A) & (a <= a[-1] - trim_A)
    if keep.sum() < 5:
        keep = np.ones_like(a, dtype=bool)
    a = a[keep]
    r = r[keep]

    sigma_samples = max(1.0, smooth_sigma_A / float(resample_step_A))
    r_smooth = gaussian_filter1d(r, sigma=sigma_samples, mode="nearest")
    return list(a), list(r_smooth)


def _normalized_profile(
    arc: List[float],
    rad: List[float],
    n_grid: int = 400,
):
    """Stretch arc to [0, 1] and interpolate radius onto a regular grid.
    Used for cross-structure comparison where tunnels have different lengths."""
    import numpy as np
    a = np.asarray(arc, dtype=float)
    r = np.asarray(rad, dtype=float)
    if a.size < 3 or a[-1] - a[0] <= 0:
        return None
    a_norm = (a - a[0]) / (a[-1] - a[0])
    grid = np.linspace(0.0, 1.0, int(n_grid))
    return np.interp(grid, a_norm, r)


def pairwise_quarter_distances(
    data: Dict[str, Dict],
    radius_field: str,
    smooth_sigma_A: float,
    trim_A: float,
    n_grid: int = 400,
    n_quarters: int = 4,
) -> Dict[Tuple[str, int], List[float]]:
    """For each pair of structures (i, j), normalize both tunnel radius profiles
    to the same [0, 1] arc grid, split into `n_quarters` equal regions, and
    compute the RMS radius difference within each region. Return a dict keyed
    by (pair_category, quarter_index) mapping to the list of pair RMS values.

    pair_category is e.g. 'Bacteria/Bacteria', 'Eukarya/Eukarya',
    'Bacteria/Eukarya'. Kingdom labels are alphabetized so 'B/E' and 'E/B'
    collapse together."""
    import numpy as np

    profiles: Dict[str, "np.ndarray"] = {}
    for sid, d in data.items():
        arc, rad = _smooth_and_trim(
            d["arc_length_A"], d[radius_field],
            smooth_sigma_A=smooth_sigma_A, trim_A=trim_A,
        )
        prof = _normalized_profile(arc, rad, n_grid=n_grid)
        if prof is not None:
            profiles[sid] = prof

    sids = sorted(profiles.keys())
    q_size = n_grid // n_quarters
    out: Dict[Tuple[str, int], List[float]] = {}

    for i, sid_a in enumerate(sids):
        kg_a = KINGDOM_BY_PDB.get(sid_a, "?").capitalize()
        for sid_b in sids[i + 1:]:
            kg_b = KINGDOM_BY_PDB.get(sid_b, "?").capitalize()
            if kg_a == kg_b:
                pair_label = f"{kg_a}/{kg_a}"
            else:
                pair_label = "/".join(sorted([kg_a, kg_b]))

            diff_sq = (profiles[sid_a] - profiles[sid_b]) ** 2
            for q in range(n_quarters):
                seg = diff_sq[q * q_size:(q + 1) * q_size]
                rms = float(np.sqrt(np.mean(seg)))
                out.setdefault((pair_label, q + 1), []).append(rms)

    return out


PAIR_COLORS = {
    "Bacteria/Bacteria": "#e74c3c",  # red
    "Eukarya/Eukarya":   "#3498db",  # blue
    "Bacteria/Eukarya":  "#8e44ad",  # purple
    "Archaea/Archaea":   "#27ae60",  # green
    "Archaea/Bacteria":  "#d35400",  # orange
    "Archaea/Eukarya":   "#16a085",  # teal
}

PAPER_PAIR_TYPES = (
    "Bacteria/Bacteria",
    "Eukarya/Eukarya",
    "Bacteria/Eukarya",
)


def plot_pairwise_distance_quarters(
    data: Dict[str, Dict],
    out_path: Path,
    radius_field: str,
    title: str,
    smooth_sigma_A: float = 2.0,
    trim_A: float = 3.0,
    pair_types: Tuple[str, ...] = PAPER_PAIR_TYPES,
    n_quarters: int = 4,
) -> Dict[Tuple[str, int], List[float]]:
    """Bar chart reproducing the paper Figure 3C style: per-quarter mean ± SEM
    RMS radius difference between structure pairs, grouped by kingdom-pair
    category. Returns the raw pair-distance dict so the caller can dump a CSV
    of every pair RMS if needed."""
    import numpy as np
    import matplotlib.pyplot as plt

    dists = pairwise_quarter_distances(
        data, radius_field=radius_field,
        smooth_sigma_A=smooth_sigma_A, trim_A=trim_A,
        n_quarters=n_quarters,
    )

    quarters = list(range(1, n_quarters + 1))
    n_pairs = len(pair_types)
    bar_w = 0.8 / n_pairs

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for pi, ptype in enumerate(pair_types):
        means, sems, counts = [], [], []
        for q in quarters:
            vals = dists.get((ptype, q), [])
            if vals:
                arr = np.asarray(vals)
                means.append(float(arr.mean()))
                sems.append(float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)
                counts.append(len(arr))
            else:
                means.append(0.0); sems.append(0.0); counts.append(0)
        x = np.asarray(quarters, dtype=float) + (pi - (n_pairs - 1) / 2.0) * bar_w
        ax.bar(
            x, means, width=bar_w, yerr=sems, capsize=4,
            color=PAIR_COLORS.get(ptype, "gray"),
            edgecolor="black", linewidth=0.5,
            label=f"{ptype} (n_pairs={counts[0] if counts else 0})",
        )

    ax.set_xlabel("Tunnel quarter region (PTC -> exit)")
    ax.set_ylabel(f"RMS radius difference (Å)")
    ax.set_xticks(quarters)
    ax.set_xticklabels([str(q) for q in quarters])
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return dists


def plot_overlay(
    data: Dict[str, Dict],
    out_path: Path,
    radius_field: str,
    title: str,
    smooth_sigma_A: float = 2.0,
    trim_A: float = 3.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for sid in sorted(data.keys()):
        d = data[sid]
        kingdom = KINGDOM_BY_PDB.get(sid, "?")
        color = KINGDOM_COLORS.get(kingdom, "gray")
        arc, rad = _smooth_and_trim(
            d["arc_length_A"], d[radius_field],
            smooth_sigma_A=smooth_sigma_A,
            trim_A=trim_A,
        )
        ax.plot(arc, rad, color=color, alpha=0.7, linewidth=1.2)

    # one legend swatch per kingdom (count actual structures plotted)
    by_king: Dict[str, int] = {}
    for s in data:
        by_king[KINGDOM_BY_PDB.get(s, "?")] = by_king.get(KINGDOM_BY_PDB.get(s, "?"), 0) + 1
    handles = [
        plt.Line2D([0], [0], color=KINGDOM_COLORS.get(k, "gray"),
                   label=f"{k.capitalize()} (n={by_king[k]})", linewidth=2.0)
        for k in KINGDOM_ORDER if by_king.get(k)
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    ax.set_xlabel("arc length from PTC (Å)")
    ax.set_ylabel(f"{radius_field.replace('_', ' ')} (Å)")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_kingdom(
    data: Dict[str, Dict],
    out_path: Path,
    radius_field: str,
    title: str,
    smooth_sigma_A: float = 2.0,
    trim_A: float = 3.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    by_king: Dict[str, List[str]] = {k: [] for k in KINGDOM_ORDER}
    for sid in data:
        k = KINGDOM_BY_PDB.get(sid, "?")
        by_king.setdefault(k, []).append(sid)

    panels = [k for k in KINGDOM_ORDER if by_king.get(k)]
    n = len(panels) or 1
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # Apply smoothing+trim before computing axis bounds so the ylim isn't
    # set by the noisy endpoint artifacts.
    smoothed: Dict[str, Tuple[List[float], List[float]]] = {}
    for sid, d in data.items():
        smoothed[sid] = _smooth_and_trim(
            d["arc_length_A"], d[radius_field],
            smooth_sigma_A=smooth_sigma_A, trim_A=trim_A,
        )

    if xlim is None:
        arc_max = max(s[0][-1] for s in smoothed.values())
        xlim = (0, arc_max * 1.02)
    if ylim is None:
        r_max = max(max(s[1]) for s in smoothed.values())
        r_min = min(min(s[1]) for s in smoothed.values())
        ylim = (max(0.0, r_min - 0.5), r_max * 1.05)

    for ax, kingdom in zip(axes, panels):
        sids = sorted(by_king[kingdom])
        color = KINGDOM_COLORS.get(kingdom, "gray")
        for sid in sids:
            arc, rad = smoothed[sid]
            ax.plot(arc, rad, color=color, alpha=0.7, linewidth=1.2, label=sid)
        ax.set_title(f"{kingdom.capitalize()} (n={len(sids)})")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.set_xlabel("arc length from PTC (Å)")
        ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=1)

    axes[0].set_ylabel(f"{radius_field.replace('_', ' ')} (Å)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb_ids", nargs="*",
                    help="PDB IDs to compare. If omitted, every run found in "
                         "the runs root is included.")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="Override the runs root (default: $NPET2_RUNS_ROOT or "
                         "~/.npet2/runs).")
    ap.add_argument("--out", type=Path, default=Path("."),
                    help="Directory to write outputs to (default: cwd).")
    ap.add_argument("--radius",
                    choices=("inscribed", "cross"),
                    default="inscribed",
                    help="Which radius column to plot. 'inscribed' = "
                         "inscribed_radius_A (MOLE probe semantics, default). "
                         "'cross' = cross_section_radius_A (perpendicular-plane).")
    ap.add_argument("--smooth", type=float, default=2.0, metavar="A",
                    help="Gaussian smoothing sigma in Angstroms applied to each "
                         "radius curve. 0 = no smoothing. Default: 2.0")
    ap.add_argument("--trim", type=float, default=3.0, metavar="A",
                    help="Trim this many Angstroms from each end of every "
                         "curve before plotting. Removes wall-hugging endpoint "
                         "artifacts from the source/sink Dijkstra voxels. Default: 3.0")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                    help="x-axis range. Default: auto-fit.")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                    help="y-axis range. Default: auto-fit (try '1 10' to match the paper).")
    ap.add_argument("--no-kingdom-panels", action="store_true",
                    help="Skip the 4-panel by-kingdom plot (only emit the overlay).")
    ap.add_argument("--no-pair-bars", action="store_true",
                    help="Skip the per-quarter pairwise-distance bar chart "
                         "(Figure 3C analog).")
    ap.add_argument("--n-quarters", type=int, default=4,
                    help="Number of equal-arc-length quarters to split each "
                         "tunnel into for the pairwise distance plot. Default: 4.")
    args = ap.parse_args(argv)

    runs_root = _runs_root(args.runs_root)
    if not runs_root.exists():
        print(f"Error: runs root {runs_root} does not exist", file=sys.stderr)
        return 1

    data = gather_centerlines(runs_root, args.pdb_ids or None)
    if not data:
        print("No centerlines found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(data)} centerlines from {runs_root}:")
    for sid in sorted(data):
        d = data[sid]
        print(f"  {sid:6s}  {KINGDOM_BY_PDB.get(sid,'?'):11s}  "
              f"n={len(d['arc_length_A']):>4d}  "
              f"len={d['arc_length_A'][-1]:6.2f}A  "
              f"min_inscribed={min(d['inscribed_radius_A']):.2f}A")

    radius_field = "inscribed_radius_A" if args.radius == "inscribed" else "cross_section_radius_A"
    label = "inscribed (MOLE-probe)" if args.radius == "inscribed" else "cross-section (perpendicular plane)"

    args.out.mkdir(parents=True, exist_ok=True)
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None

    plot_overlay(
        data, args.out / "centerline_overlay.png",
        radius_field=radius_field,
        title=f"Exit-tunnel radius profiles ({label})",
        smooth_sigma_A=args.smooth,
        trim_A=args.trim,
        xlim=xlim, ylim=ylim,
    )
    if not args.no_kingdom_panels:
        plot_by_kingdom(
            data, args.out / "centerline_by_kingdom.png",
            radius_field=radius_field,
            title=f"Exit-tunnel radius profiles by kingdom ({label})",
            smooth_sigma_A=args.smooth,
            trim_A=args.trim,
            xlim=xlim, ylim=ylim,
        )
    pair_dists = None
    if not args.no_pair_bars:
        pair_dists = plot_pairwise_distance_quarters(
            data, args.out / "centerline_pairwise_distance.png",
            radius_field=radius_field,
            title=f"Pairwise tunnel radius distance by quarter ({label})",
            smooth_sigma_A=args.smooth,
            trim_A=args.trim,
            n_quarters=args.n_quarters,
        )
        # Also dump the raw per-pair distances so the user can reanalyze.
        import numpy as np
        with (args.out / "centerline_pairwise_distance.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_category", "quarter", "n_pairs", "mean_rmsd_A", "sem_rmsd_A", "values_A"])
            for (cat, q), vals in sorted(pair_dists.items()):
                arr = np.asarray(vals, dtype=float)
                sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                w.writerow([
                    cat, q, len(arr),
                    f"{arr.mean():.4f}", f"{sem:.4f}",
                    ";".join(f"{v:.4f}" for v in arr),
                ])
    write_summary_csv(data, args.out / "centerline_summary.csv", radius_field)

    print(f"\nWrote:")
    print(f"  {args.out / 'centerline_overlay.png'}")
    if not args.no_kingdom_panels:
        print(f"  {args.out / 'centerline_by_kingdom.png'}")
    if not args.no_pair_bars:
        print(f"  {args.out / 'centerline_pairwise_distance.png'}")
        print(f"  {args.out / 'centerline_pairwise_distance.csv'}")
    print(f"  {args.out / 'centerline_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
