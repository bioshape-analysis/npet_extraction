[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rtviii/npet_extraction)

# NPET Extraction Pipeline

Pipeline to extract a **watertight mesh** of the ribosome **nascent peptide exit tunnel (NPET)** and compute a **tunnel-lining report** from an RCSB structure ID (or local mmCIF). Defaults: **mmCIF from RCSB**, **profile + landmarks from ribosome.xyz**.

<p align="center">
  <img src="docs/grid0.5.gif" width="49%" alt="NPET2 grid refine (0.5Å)" />
  <img src="docs/grid1.0.gif" width="49%" alt="NPET2 coarse grid (1.0Å)" />
</p>
<p align="center"><sub>
  Tunnel mesh of the 5NWY structure. Left: 0.5Å voxel grid; Right: 1Å voxel grid, smoothed.
</sub></p>

<details>
  <summary><strong>
You are finally here? Great! Cite us.
  </strong></summary>

```bibtex

@article{yu2026advanced,
  title={Advanced coarse-grained model for fast simulation of nascent polypeptide chain dynamics within the ribosome},
  author={Yu, Shiqi and Kushner, Artem and Teasell, Ella and Zhao, Wenjun and Srebnik, Simcha and Duc, Khanh Dao},
  journal={Biophysical Journal},
  volume={125},
  number={2},
  pages={641--651},
  year={2026},
  publisher={Elsevier}
}
```
</details>

---

## Install (local)

```bash
git clone https://github.com/rtviii/npet_extraction
cd npet_extraction

python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools
pip install .
```

### Simply run this (local; ribosome.xyz for deps/landmarks)

```bash
npet2 run 5NWY
```

---

## Install & run (Docker)

### Build

```bash
git clone https://github.com/rtviii/npet_extraction
cd npet_extraction
docker build -t npet2:latest .
```

### Simply run this (Docker; ribosome.xyz for deps/landmarks)

```bash
mkdir -p ./npet2_data
docker run --rm -it \
  -v "$(pwd)/npet2_data:/data" \
  npet2:latest run 5NWY
```

### docker compose

```bash
docker compose run --rm npet2 run 5NWY
```

---

## Environment variables

Defined in [`libnpet/core/config.py`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/core/config.py)

| Variable                  |                Default (local) |           Default (docker) | Meaning                                        |
| ------------------------- | -----------------------------: | -------------------------: | ---------------------------------------------- |
| `NPET2_ROOT`              |                     `~/.npet2` |                    `/data` | Root for all data                              |
| `NPET2_RUNS_ROOT`         |             `$NPET2_ROOT/runs` |               `/data/runs` | **Output root** (where runs/logs/artifacts go) |
| `NPET2_CACHE_ROOT`        |            `$NPET2_ROOT/cache` |              `/data/cache` | Stage cache                                    |
| `NPET2_POISSON_RECON_BIN` | `$NPET2_ROOT/bin/PoissonRecon` |   `/data/bin/PoissonRecon` | Optional external binary                       |
| `NPET2_RIBOXYZ_API_URL`   |        `http://localhost:8000` | `https://api.ribosome.xyz` | ribosome.xyz API base URL                      |

---

## Output structure

A run is written under:
`{NPET2_RUNS_ROOT}/{RCSB_ID}/{RUN_ID}/`

Key files:

* `manifest.json` (run ledger + artifact index) — [`libnpet/core/manifest.py`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/core/manifest.py)
* `tunnel_mesh.ply` / `tunnel_mesh_ascii.ply` (final watertight tunnel mesh)
* `tunnel_lining.json` + `tunnel_lining.cif` (lining report + extracted chains)

Stage directories:

* `stage/00_inputs/` — atoms loaded
* `stage/10_landmarks/` — PTC + constriction
* `stage/20_exterior_shell/` — exterior shell mesh
* `stage/30_region_atoms/` — cylinder-filtered “wall” atoms
* `stage/40_empty_space/` — empty voxel centers (inside shell & cylinder)
* `stage/50_clustering/` — DBSCAN coarse/refine + optional level_0 mesh
* `stage/55_grid_refine/` — ROI refine + optional level_1 mesh
* `stage/70_mesh_validate/` — final mesh selection/validation
* `stage/80_tunnel_lining/` — lining report + lining mmCIF

(See pipeline composition in [`libnpet/run.py`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/run.py).)

---

## Dependency inputs: local files vs ribosome.xyz API (default)

Provider logic: [`libnpet/adapters/standalone_providers.py`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/adapters/standalone_providers.py)

### Defaults (no files)

* mmCIF: downloaded from RCSB + cached under `${NPET2_ROOT}/mmcif/{RCSB_ID}/{RCSB_ID}.cif`
* profile: ribosome.xyz API
* landmarks (PTC + constriction): ribosome.xyz API

### Override with local files

```bash
# local mmCIF, API for profile + landmarks
npet2 run 5NWY --mmcif /path/to/5NWY.cif

# all local
npet2 run 5NWY \
  --mmcif /path/to/5NWY.cif \
  --profile /path/to/5NWY_profile.json \
  --ptc /path/to/5NWY_PTC.json \
  --constriction /path/to/5NWY_CONSTRICTION_SITE.json
```

### Schemas (must match)

**PTC JSON** (validated by [`PTCInfo`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/core/ribosome_types.py))

```json
{"location":[x,y,z]}
```

**Constriction JSON** (validated by [`ConstrictionInfo`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/core/ribosome_types.py))

```json
{"location":[x,y,z]}
```

**Profile JSON** (validated by [`RibosomeProfile`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/core/ribosome_types.py))

* must include at least: `rcsb_id`
* optional: `mitochondrial`, `proteins`, `rnas`, `other_polymers`, `assembly_map`

Minimal example:

```json
{
  "rcsb_id": "5NWY",
  "mitochondrial": false,
  "proteins": [{"auth_asym_id":"A","assembly_id":0,"nomenclature":[],"rcsb_pdbx_description":null,"entity_poly_seq_length":0}],
  "rnas": [{"auth_asym_id":"r","assembly_id":0,"nomenclature":[],"rcsb_pdbx_description":null,"entity_poly_seq_length":0}],
  "other_polymers": [],
  "assembly_map": [
    {"rcsb_id":"5NWY","polymer_entity_instances":[{"entity_id":"1","auth_asym_id":"A"}],"nonpolymer_entity_instances":null}
  ]
}
```

---

## CLI options

CLI implementation: [`libnpet/__main__.py`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/__main__.py)

### Commands

* `npet2 setup` — build/install external binaries (PoissonRecon) under `$NPET2_ROOT/bin/`
* `npet2 show-config` — print default [`RunConfig`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/core/config.py) as JSON
* `npet2 run` — run the pipeline

### `npet2 run` flags (what they control)

**Targets**

* `RCSB_ID ...` / `--rcsb_id ID` / `--from-file FILE` — which structures to process

**Inputs / providers**

* `--api-url URL` — override ribosome.xyz base URL (else `NPET2_RIBOXYZ_API_URL`)
* `--mmcif PATH` — use local mmCIF (else RCSB download)
* `--profile PATH` — use local profile JSON (else API)
* `--ptc PATH` — use local PTC JSON (else API)
* `--constriction PATH` — use local constriction JSON (else API)
* `--data-dir DIR` — override `$NPET2_ROOT` (mmCIF cache/build defaults)

**Outputs / parallelism**

* `--output-dir DIR` — override `$NPET2_RUNS_ROOT`
* `--workers, -j N` — parallel runs across multiple IDs

**Config overrides** (maps into [`RunConfig`](https://github.com/rtviii/npet_extraction/blob/main/libnpet/core/config.py))

* `--config-json PATH` — full RunConfig JSON
* `--cylinder-radius A`, `--cylinder-height A`, `--ptc-extension A` — cylinder region definition (Stages 10/30/40)
* `--voxel-size A` — level_0 grid voxel size (Stage 40 → Stage 50 density)
* `--refine-voxel-size A` — refinement voxel size (Stage 55)
* `--dbscan-coarse-eps A`, `--dbscan-coarse-min-samples N` — Stage 50 coarse DBSCAN
* `--dbscan-refine-eps A`, `--dbscan-refine-min-samples N` — Stage 50 refine DBSCAN
* `--no-mesh` — disable level_0/level_1 meshing (Stage 50/55). (Stage 70 needs a mesh.)
* `--no-refine` — skip Stage 55 refinement (if honored by your pipeline build)
* `--lining-proximity A` — Stage 80 mesh→atom proximity threshold
* `--no-nonpolymers` — Stage 80 exclude nonpolymers
* `--include-waters` — Stage 80 include water chains



