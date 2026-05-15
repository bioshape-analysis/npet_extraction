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


def plot_overlay(
    data: Dict[str, Dict],
    out_path: Path,
    radius_field: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    for sid in sorted(data.keys()):
        d = data[sid]
        kingdom = KINGDOM_BY_PDB.get(sid, "?")
        color = KINGDOM_COLORS.get(kingdom, "gray")
        ax.plot(d["arc_length_A"], d[radius_field],
                color=color, alpha=0.7, linewidth=1.0)
        # label at the rightmost point of each curve
        ax.annotate(
            sid,
            xy=(d["arc_length_A"][-1], d[radius_field][-1]),
            xytext=(3, 0), textcoords="offset points",
            fontsize=6, color=color, va="center",
        )

    # legend swatches by kingdom
    handles = [
        plt.Line2D([0], [0], color=col, label=k.capitalize(), linewidth=2.0)
        for k, col in KINGDOM_COLORS.items()
        if any(KINGDOM_BY_PDB.get(s) == k for s in data.keys())
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    ax.set_xlabel("arc length from PTC (A)")
    ax.set_ylabel(f"{radius_field} (A)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_kingdom(
    data: Dict[str, Dict],
    out_path: Path,
    radius_field: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    by_king: Dict[str, List[str]] = {k: [] for k in KINGDOM_ORDER}
    for sid in data:
        k = KINGDOM_BY_PDB.get(sid, "?")
        by_king.setdefault(k, []).append(sid)

    panels = [k for k in KINGDOM_ORDER if by_king.get(k)]
    n = len(panels) or 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # global axis bounds for parity
    arc_max = max(max(d["arc_length_A"]) for d in data.values())
    r_max = max(max(d[radius_field]) for d in data.values())

    for ax, kingdom in zip(axes, panels):
        sids = sorted(by_king[kingdom])
        color = KINGDOM_COLORS.get(kingdom, "gray")
        for sid in sids:
            d = data[sid]
            ax.plot(d["arc_length_A"], d[radius_field],
                    color=color, alpha=0.6, linewidth=1.0, label=sid)
        ax.set_title(f"{kingdom.capitalize()} (n={len(sids)})")
        ax.set_xlim(0, arc_max * 1.05)
        ax.set_ylim(0, r_max * 1.05)
        ax.grid(alpha=0.3)
        ax.set_xlabel("arc length from PTC (A)")
        ax.legend(fontsize=6, loc="upper right", frameon=False)

    axes[0].set_ylabel(f"{radius_field} (A)")
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
    plot_overlay(
        data, args.out / "centerline_overlay.png",
        radius_field=radius_field,
        title=f"Exit-tunnel radius profiles ({label})",
    )
    plot_by_kingdom(
        data, args.out / "centerline_by_kingdom.png",
        radius_field=radius_field,
        title=f"Exit-tunnel radius profiles by kingdom ({label})",
    )
    write_summary_csv(data, args.out / "centerline_summary.csv", radius_field)

    print(f"\nWrote:")
    print(f"  {args.out / 'centerline_overlay.png'}")
    print(f"  {args.out / 'centerline_by_kingdom.png'}")
    print(f"  {args.out / 'centerline_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
