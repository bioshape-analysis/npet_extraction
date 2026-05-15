# libnpet/__main__.py
"""
npet2 - Ribosome nascent peptide exit tunnel geometry pipeline

Two modes:

  API mode   -- pass an RCSB ID, everything is fetched automatically.
                mmCIF is downloaded from RCSB, profile/landmarks from ribosome.xyz.

    npet2 run 5NWY --api-url http://localhost:8000

  File mode  -- point at local files explicitly.
                Any omitted file falls back to the API.

    npet2 run 5NWY \\
        --mmcif   /data/5NWY.cif \\
        --profile /data/5NWY.json \\
        --ptc     /data/5NWY_PTC.json \\
        --constriction /data/5NWY_CONSTRICTION_SITE.json

  Mixed      -- local mmCIF, API for the rest (useful when you have the
                structure locally but not the landmarks yet)

    npet2 run 5NWY --mmcif /data/5NWY.cif --api-url http://localhost:8000

Advanced config overrides are available via --config-json or individual flags.
Run `npet2 show-config` to see all defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="npet2",
        description="Ribosome exit tunnel geometry pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command")

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    setup_p = sub.add_parser("setup", help="Build required external binaries (PoissonRecon)")
    setup_p.add_argument("--data-dir", metavar="DIR",
                         help="Root directory for npet2 data (default: ~/.npet2 or $NPET2_ROOT)")

    # -------------------------------------------------------------------------
    # run
    # -------------------------------------------------------------------------
    run_p = sub.add_parser(
        "run",
        help="Run the pipeline on one or more structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target structures
    id_grp = run_p.add_argument_group("target structures (provide at least one)")
    id_grp.add_argument("rcsb_ids", nargs="*", metavar="RCSB_ID",
                        help="One or more RCSB IDs (e.g. 5NWY 7K00)")
    id_grp.add_argument("--rcsb_id", dest="rcsb_id_flag", metavar="RCSB_ID",
                        help="Single RCSB ID (alternative to positional)")
    id_grp.add_argument("--from-file", metavar="FILE",
                        help="Text file with one RCSB ID per line")

    # Data sources
    src_grp = run_p.add_argument_group(
        "data sources",
        "All inputs default to the ribosome.xyz API when not provided as files.\n"
        "The mmCIF is downloaded from RCSB if --mmcif is omitted.",
    )
    src_grp.add_argument("--api-url", metavar="URL", default=None,
                         help="ribosome.xyz API base URL (default: $NPET2_RIBOXYZ_API_URL "
                              "or http://localhost:8000)")
    src_grp.add_argument("--mmcif", metavar="PATH",
                         help="mmCIF file path. If omitted, downloaded from RCSB and cached.")
    src_grp.add_argument("--profile", metavar="PATH",
                         help="RibosomeStructure JSON. If omitted, fetched from API.")
    src_grp.add_argument("--ptc", metavar="PATH",
                         help="PTC landmark JSON {\"location\":[x,y,z],...}. "
                              "If omitted, fetched from API.")
    src_grp.add_argument("--constriction", metavar="PATH",
                         help="Constriction site JSON {\"location\":[x,y,z]}. "
                              "If omitted, fetched from API.")
    src_grp.add_argument("--data-dir", metavar="DIR",
                         help="Root directory for all npet2 data: mmcif cache, runs, build "
                              "(default: ~/.npet2 or $NPET2_ROOT)")

    # Output
    run_p.add_argument("--output-dir", metavar="DIR",
                       help="Root directory for run outputs (default: $NPET2_RUNS_ROOT)")

    # Parallelism
    run_p.add_argument("--workers", "-j", type=int, default=1, metavar="N",
                       help="Parallel workers for batch runs (default: 1)")

    # Advanced config overrides
    cfg = run_p.add_argument_group(
        "advanced config overrides",
        "Override individual pipeline parameters. "
        "Use --config-json to supply a full RunConfig JSON instead."
    )
    cfg.add_argument("--config-json", metavar="PATH",
                     help="Path to a RunConfig JSON (see `npet2 show-config`)")
    cfg.add_argument("--cylinder-radius", type=float, metavar="A")
    cfg.add_argument("--cylinder-height", type=float, metavar="A")
    cfg.add_argument("--ptc-extension", type=float, metavar="A")
    cfg.add_argument("--voxel-size", type=float, metavar="A",
                     help="Level-0 grid voxel size in Angstroms")
    cfg.add_argument("--refine-voxel-size", type=float, metavar="A")
    cfg.add_argument("--no-mesh", action="store_true", help="Disable mesh generation")
    cfg.add_argument("--no-refine", action="store_true", help="Skip Stage55 grid refinement")
    cfg.add_argument("--dbscan-coarse-eps", type=float, metavar="A")
    cfg.add_argument("--dbscan-coarse-min-samples", type=int)
    cfg.add_argument("--dbscan-refine-eps", type=float, metavar="A")
    cfg.add_argument("--dbscan-refine-min-samples", type=int)

    cfg.add_argument("--lining-proximity", type=float, metavar="A",
                    help="Atom-to-mesh proximity threshold for tunnel lining (default: 5.0A)")
    cfg.add_argument("--no-nonpolymers", action="store_true",
                    help="Exclude nonpolymer ligands/ions from lining output")
    cfg.add_argument("--include-waters", action="store_true",
                    help="Include water molecules in lining output")
    # -------------------------------------------------------------------------
    # centerline (standalone: re-run Stage60 on an existing run)
    # -------------------------------------------------------------------------
    cl_p = sub.add_parser(
        "centerline",
        help="Extract a MOLE-style centerline + radius CSV from an existing run",
        description=(
            "Runs Stage60 (centerline extraction) on an existing run directory. "
            "Re-runs only the centerline step; the rest of the pipeline output is preserved. "
            "Use this to sweep --r-open or --alpha without re-running the whole pipeline."
        ),
    )
    cl_p.add_argument("rcsb_id", metavar="RCSB_ID",
                      help="RCSB ID of the structure (e.g. 5NWY)")
    cl_p.add_argument("--run-id", metavar="RUN_ID",
                      help="Specific run_id under <runs_root>/<RCSB_ID>/. "
                           "If omitted, uses the most-recently-modified run.")
    cl_p.add_argument("--data-dir", metavar="DIR",
                      help="Root directory for npet2 data (default: ~/.npet2 or $NPET2_ROOT)")
    cl_p.add_argument("--output-dir", metavar="DIR",
                      help="Root directory containing runs (default: $NPET2_RUNS_ROOT)")
    cl_p.add_argument("--r-open", type=float, metavar="A", default=None,
                      help="Side-channel pruning radius in Angstroms (default: 1.0)")
    cl_p.add_argument("--alpha", type=float, metavar="N", default=None,
                      help="Dijkstra cost weight for clearance bias (default: 1.0)")
    cl_p.add_argument("--resample-step", type=float, metavar="A", default=None,
                      help="Uniform arc-length resample step in Angstroms (default: 0.5)")
    cl_p.add_argument("--smoothing", type=float, metavar="S", default=None,
                      help="Spline smoothing factor (default: 0.0 = pass through)")
    cl_p.add_argument("--n-rays", type=int, metavar="N", default=None,
                      help="Rays per perpendicular plane for cross-section radius (default: 64)")

    # -------------------------------------------------------------------------
    # show-config
    # -------------------------------------------------------------------------
    sub.add_parser("show-config", help="Print the default RunConfig as JSON")

    return p


def _build_config(args) -> "RunConfig":
    from libnpet.core.config import RunConfig, GridLevelConfig

    if args.config_json:
        data = json.loads(Path(args.config_json).read_text())
        if "grid_levels" in data:
            data["grid_levels"] = [GridLevelConfig(**gl) for gl in data["grid_levels"]]
        return RunConfig(**data)

    kwargs = {}
    if args.cylinder_radius is not None:
        kwargs["cylinder_radius_A"] = args.cylinder_radius
    if args.cylinder_height is not None:
        kwargs["cylinder_height_A"] = args.cylinder_height
    if args.ptc_extension is not None:
        kwargs["cylinder_ptc_extension_A"] = args.ptc_extension
    if args.refine_voxel_size is not None:
        kwargs["refine_voxel_size_A"] = args.refine_voxel_size
    if args.no_mesh:
        kwargs["mesh_level0_enable"] = False
        kwargs["mesh_level1_enable"] = False
    if args.dbscan_coarse_eps is not None:
        kwargs["dbscan_level0_coarse_eps_A"] = args.dbscan_coarse_eps
    if args.dbscan_coarse_min_samples is not None:
        kwargs["dbscan_level0_coarse_min_samples"] = args.dbscan_coarse_min_samples
    if args.dbscan_refine_eps is not None:
        kwargs["dbscan_level0_refine_eps_A"] = args.dbscan_refine_eps
    if args.dbscan_refine_min_samples is not None:
        kwargs["dbscan_level0_refine_min_samples"] = args.dbscan_refine_min_samples
    if args.voxel_size is not None:
        kwargs["grid_levels"] = [
            GridLevelConfig(name="level_0", voxel_size_A=args.voxel_size,
                            occupancy_backend="legacy_kdtree"),
        ]
    if args.lining_proximity is not None:
        kwargs["lining_proximity_A"] = args.lining_proximity
    if args.no_nonpolymers:
        kwargs["lining_include_nonpolymers"] = False
    if args.include_waters:
        kwargs["lining_include_waters"] = True
    return RunConfig(**kwargs)


def _collect_rcsb_ids(args) -> list[str]:
    ids = list(args.rcsb_ids) if args.rcsb_ids else []
    if getattr(args, "rcsb_id_flag", None):
        ids.append(args.rcsb_id_flag)
    if args.from_file:
        p = Path(args.from_file)
        if not p.exists():
            print(f"Error: --from-file {p} does not exist", file=sys.stderr)
            sys.exit(1)
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    if not ids:
        print("Error: no RCSB IDs specified.", file=sys.stderr)
        print("  Provide as positional args, --rcsb_id, or --from-file.", file=sys.stderr)
        sys.exit(1)
    return [x.upper() for x in ids]


def _make_providers(args, rcsb_id: str):
    from libnpet.adapters.standalone_providers import (
        FileStructureProvider,
        FileLandmarkProvider,
        _download_mmcif,
    )
    from libnpet.core.config import SETTINGS

    api_base = args.api_url or SETTINGS.riboxyz_api_base

    # mmCIF: explicit path, or auto-download from RCSB
    if args.mmcif:
        mmcif_path = Path(args.mmcif)
        if not mmcif_path.exists():
            print(f"Error: --mmcif {mmcif_path} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        mmcif_path = _download_mmcif(rcsb_id)

    profile_path = Path(args.profile) if args.profile else None
    ptc_path     = Path(args.ptc)     if args.ptc     else None
    constr_path  = Path(args.constriction) if args.constriction else None

    sp = FileStructureProvider(
        mmcif_path=mmcif_path,
        profile_path=profile_path,
        api_base=api_base,
    )
    lp = FileLandmarkProvider(
        ptc_path=ptc_path,
        constriction_path=constr_path,
        api_base=api_base,
    )
    return sp, lp


def _apply_data_dir(args) -> None:
    """Override SETTINGS.npet2_root and derived paths if --data-dir was given."""
    data_dir = getattr(args, "data_dir", None)
    if not data_dir:
        return
    import libnpet.core.config as cfg_mod
    root = Path(data_dir)
    cfg_mod.SETTINGS = cfg_mod.Settings(
        npet2_root        = root,
        runs_root         = root / "runs",
        cache_root        = root / "cache",
        poisson_recon_bin = str(root / "bin" / "PoissonRecon"),
        riboxyz_api_base  = cfg_mod.SETTINGS.riboxyz_api_base,
    )
def _run_single(rcsb_id: str, args, config, output_root: Optional[Path]) -> dict:
    from libnpet.run import run_npet2

    _apply_data_dir(args) 
    try:
        sp, lp = _make_providers(args, rcsb_id)

        if output_root:
            import libnpet.core.config as cfg_mod
            cfg_mod.SETTINGS = cfg_mod.Settings(
                runs_root=output_root,
                npet2_root=cfg_mod.SETTINGS.npet2_root,
                cache_root=cfg_mod.SETTINGS.cache_root,
                poisson_recon_bin=cfg_mod.SETTINGS.poisson_recon_bin,
                riboxyz_api_base=cfg_mod.SETTINGS.riboxyz_api_base,
            )

        ctx = run_npet2(rcsb_id, config, structure_provider=sp, landmark_provider=lp)
        return {"rcsb_id": rcsb_id, "status": "success", "run_dir": str(ctx.store.run_dir)}
    except Exception as e:
        return {
            "rcsb_id": rcsb_id,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _run_worker(packed_args: tuple) -> dict:
    rcsb_id, args_ns, config_dict, output_root_str = packed_args
    from libnpet.core.config import RunConfig, GridLevelConfig

    if "grid_levels" in config_dict:
        config_dict["grid_levels"] = [GridLevelConfig(**gl) for gl in config_dict["grid_levels"]]
    config = RunConfig(**config_dict)
    output_root = Path(output_root_str) if output_root_str else None
    return _run_single(rcsb_id, args_ns, config, output_root)

def _cmd_setup(args=None):
    import platform
    import shutil
    import subprocess

    if args:
        _apply_data_dir(args)

    from libnpet.core.config import SETTINGS

    dest = Path(SETTINGS.poisson_recon_bin)

    print("npet2 setup: checking for PoissonRecon...")
    if dest.exists():
        print(f"  already built: {dest}")
        return

    system = platform.system()

    # prerequisites
    missing = []
    for tool in ("git", "make"):
        if not shutil.which(tool):
            missing.append(tool)
    if not (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")):
        missing.append("c++ compiler (g++ or clang++)")

    if missing:
        print("  missing required tools:", file=sys.stderr)
        for m in missing:
            print(f"    - {m}", file=sys.stderr)
        print(
            "\n  Install them and retry.\n"
            "\n  macOS:\n"
            "    xcode-select --install\n"
            "\n  Ubuntu/Debian:\n"
            "    apt-get install -y git build-essential",
            file=sys.stderr,
        )
        sys.exit(1)

    # libjpeg: required by PoissonRecon's image I/O
    if system == "Darwin":
        brew = shutil.which("brew")
        # check if jpeg headers are findable
        import subprocess as sp
        jpeg_ok = sp.run(
            ["clang++", "-x", "c++", "-include", "jpeglib.h", "-E", "-"],
            input="", capture_output=True, text=True
        ).returncode == 0
        if not jpeg_ok:
            if brew:
                print("  libjpeg not found, installing via brew ...")
                subprocess.run(["brew", "install", "jpeg"], check=True)
            else:
                print(
                    "  libjpeg not found and brew is not available.\n"
                    "  Install brew (https://brew.sh) then run: brew install jpeg",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        # on Linux check for the header directly
        jpeg_header = Path("/usr/include/jpeglib.h")
        if not jpeg_header.exists():
            print(
                "  libjpeg-dev not found.\n"
                "  Install it with: apt-get install -y libjpeg-dev",
                file=sys.stderr,
            )
            sys.exit(1)

    build_root = Path(SETTINGS.npet2_root) / "build" / "PoissonRecon"
    build_root.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    repo_dir = build_root / "src"
    if not repo_dir.exists():
        print("  cloning mkazhdan/PoissonRecon ...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/mkazhdan/PoissonRecon.git", str(repo_dir)],
            check=True,
        )
    else:
        print("  source already cloned, skipping git clone")

    if system == "Darwin":
        compiler = "clang"
        print("  macOS detected: building with COMPILER=clang (no OpenMP)")
        # brew jpeg lives under a non-standard prefix on Apple Silicon / Intel
        brew_prefix = subprocess.run(
            ["brew", "--prefix", "jpeg"],
            capture_output=True, text=True
        ).stdout.strip()
        extra_env = {
            **__import__("os").environ,
            "CPATH": f"{brew_prefix}/include",
            "LIBRARY_PATH": f"{brew_prefix}/lib",
        }
    else:
        compiler = "gcc"
        print("  Linux detected: building with COMPILER=gcc (OpenMP enabled)")
        extra_env = None

    print("  building PoissonRecon (this takes a minute) ...")
    subprocess.run(
        ["make", "-j4", "poissonrecon", f"COMPILER={compiler}"],
        cwd=str(repo_dir),
        env=extra_env,
        check=True,
    )

    candidates = [c for c in (repo_dir / "Bin").rglob("PoissonRecon") if c.is_file()]
    if not candidates:
        print("  build finished but PoissonRecon binary not found", file=sys.stderr)
        print(f"  look in {repo_dir / 'Bin'} and set NPET2_POISSON_RECON_BIN manually",
              file=sys.stderr)
        sys.exit(1)

    shutil.copy2(str(candidates[0]), str(dest))
    dest.chmod(0o755)
    print(f"  PoissonRecon installed to {dest}")
    print(f"  all npet2 data will live under {SETTINGS.npet2_root}")
    print("  setup complete. You can now run: npet2 run <RCSB_ID>")


def _cmd_centerline(args) -> None:
    """Re-run Stage60Centerline on an existing run directory."""
    _apply_data_dir(args)
    import libnpet.core.config as cfg_mod
    from libnpet.core.manifest import RunManifest
    from libnpet.core.store import LocalRunStore
    from libnpet.core.types import StageContext
    from libnpet.core.config import RunConfig
    from libnpet.stages.centerline import Stage60Centerline

    output_root = Path(args.output_dir) if args.output_dir else cfg_mod.SETTINGS.runs_root
    rcsb_id = args.rcsb_id.upper()
    struct_runs_dir = output_root / rcsb_id

    if not struct_runs_dir.exists():
        print(f"Error: no runs found at {struct_runs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.run_id:
        run_dir = struct_runs_dir / args.run_id
        if not run_dir.exists():
            print(f"Error: run {run_dir} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        candidates = [d for d in struct_runs_dir.iterdir() if d.is_dir()]
        if not candidates:
            print(f"Error: no runs under {struct_runs_dir}", file=sys.stderr)
            sys.exit(1)
        run_dir = max(candidates, key=lambda d: d.stat().st_mtime)
        print(f"npet2 centerline: using most-recent run {run_dir.name}")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: no manifest.json at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    grid_spec_path = run_dir / "stage" / "55_grid_refine" / "grid_spec_level_1.json"
    mask_path = run_dir / "stage" / "55_grid_refine" / "selected_void_component_mask_level_1.npy"
    if not grid_spec_path.exists() or not mask_path.exists():
        print(
            f"Error: Stage55 artifacts not found under {run_dir}.\n"
            f"  expected {grid_spec_path}\n"
            f"  expected {mask_path}\n"
            f"Run the full pipeline first: npet2 run {rcsb_id}",
            file=sys.stderr,
        )
        sys.exit(1)

    grid_spec = json.loads(grid_spec_path.read_text())
    ptc_world = grid_spec["transform"]["ptc"]
    constr_world = grid_spec["transform"]["constriction"]

    cfg_overrides = {}
    if args.r_open is not None:
        cfg_overrides["centerline_open_radius_A"] = args.r_open
    if args.alpha is not None:
        cfg_overrides["centerline_path_alpha"] = args.alpha
    if args.resample_step is not None:
        cfg_overrides["centerline_resample_step_A"] = args.resample_step
    if args.smoothing is not None:
        cfg_overrides["centerline_smoothing"] = args.smoothing
    if args.n_rays is not None:
        cfg_overrides["centerline_n_radial_rays"] = args.n_rays
    config = RunConfig(**cfg_overrides)

    manifest = RunManifest.from_path(manifest_path)
    store = LocalRunStore(run_dir=run_dir, manifest=manifest)
    ctx = StageContext(
        run_id=manifest.run_id,
        rcsb_id=rcsb_id,
        config=config,
        store=store,
        inputs={
            "selected_void_component_mask_level_1_path": str(mask_path),
            "grid_spec_level_1_path": str(grid_spec_path),
            "ptc_xyz": ptc_world,
            "constriction_xyz": constr_world,
        },
    )

    stage = Stage60Centerline()
    params = stage.params(ctx)
    store.begin_stage(stage.key, params=params)
    print(f"npet2 centerline: run_dir={run_dir}")
    print(f"  params: {params}")
    try:
        stage.run(ctx)
        store.end_stage(stage.key, success=True)
    except Exception as e:
        store.end_stage(stage.key, success=False, note=f"err={e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"npet2 centerline: done")
    print(f"  CSV: {run_dir / 'stage' / '60_centerline' / 'centerline.csv'}")
    print(f"  PLY: {run_dir / 'stage' / '60_centerline' / 'centerline.ply'}")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "setup":
        _cmd_setup(args)
        return

    if args.command == "show-config":
        from libnpet.core.config import RunConfig
        print(json.dumps(asdict(RunConfig()), indent=2))
        return

    if args.command == "centerline":
        _cmd_centerline(args)
        return

    if args.command == "run":
        _apply_data_dir(args)
        rcsb_ids    = _collect_rcsb_ids(args)
        config      = _build_config(args)
        output_root = Path(args.output_dir) if args.output_dir else None
        n_workers   = min(args.workers, len(rcsb_ids))

        print(f"npet2: processing {len(rcsb_ids)} structure(s), workers={n_workers}")

        if n_workers <= 1:
            results = []
            for rid in rcsb_ids:
                r = _run_single(rid, args, config, output_root)
                results.append(r)
                status = r["status"]
                print(f"  {rid}: {status}" + (
                    f" -> {r.get('run_dir', '')}" if status == "success"
                    else f"\n{r.get('traceback', r.get('error', ''))}"
                ))
        else:
            config_dict     = asdict(config)
            output_root_str = str(output_root) if output_root else None
            packed = [(rid, args, config_dict, output_root_str) for rid in rcsb_ids]
            results = []
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_worker, p): p[0] for p in packed}
                for fut in as_completed(futures):
                    rid = futures[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        r = {"rcsb_id": rid, "status": "failed", "error": str(e)}
                    results.append(r)
                    status = r["status"]
                    print(f"  {rid}: {status}" + (
                        f" -> {r.get('run_dir', '')}" if status == "success"
                        else f" ({r.get('error', '')})"
                    ))

        ok   = sum(1 for r in results if r["status"] == "success")
        fail = len(results) - ok
        print(f"\nnpet2: {ok} succeeded, {fail} failed out of {len(results)}")
        if fail > 0:
            sys.exit(1)



if __name__ == "__main__":
    main()