"""Batch-run helpers for `npet2 run`.

These live in their own module (not in `libnpet/__main__.py`) so that when
ProcessPoolExecutor uses Python's `spawn` start method (default on macOS),
the worker subprocess can pickle/unpickle the worker function by importing
it as `libnpet._batch.run_worker`. If these functions lived in __main__.py
the subprocess would fail to look them up — its own `__main__` is the
spawn bootstrap, not `libnpet/__main__.py`.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional


def apply_data_dir(args) -> None:
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


def make_providers(args, rcsb_id: str):
    from libnpet.adapters.standalone_providers import (
        FileStructureProvider,
        FileLandmarkProvider,
        _download_mmcif,
    )
    from libnpet.core.config import SETTINGS

    api_base = args.api_url or SETTINGS.riboxyz_api_base

    if args.mmcif:
        mmcif_path = Path(args.mmcif)
        if not mmcif_path.exists():
            print(f"Error: --mmcif {mmcif_path} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        mmcif_path = _download_mmcif(rcsb_id)

    profile_path = Path(args.profile)       if args.profile else None
    ptc_path     = Path(args.ptc)           if args.ptc     else None
    constr_path  = Path(args.constriction)  if args.constriction else None

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


def run_single(rcsb_id: str, args, config, output_root: Optional[Path]) -> dict:
    from libnpet.run import run_npet2

    apply_data_dir(args)
    try:
        sp, lp = make_providers(args, rcsb_id)

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


def run_worker(packed_args: tuple) -> dict:
    """Entry point for ProcessPoolExecutor. Must live at module top level
    so the subprocess can pickle it as `libnpet._batch.run_worker`."""
    rcsb_id, args_ns, config_dict, output_root_str = packed_args
    from libnpet.core.config import RunConfig, GridLevelConfig

    if "grid_levels" in config_dict:
        config_dict["grid_levels"] = [GridLevelConfig(**gl) for gl in config_dict["grid_levels"]]
    config = RunConfig(**config_dict)
    output_root = Path(output_root_str) if output_root_str else None
    return run_single(rcsb_id, args_ns, config, output_root)
