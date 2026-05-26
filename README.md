# allen-mouse-connectivity

Build a mouse brain structural connectivity matrix from the [Allen Institute
Mouse Connectivity Atlas](http://connectivity.brain-map.org/).

**No TVB required.**  Just `numpy` and `allensdk`.

`allen-mouse-connectivity` is a standalone command-line tool extracted from
[The Virtual Brain](https://www.thevirtualbrain.org) (`tvb-mouse` monorepo).
The original code was embedded inside TVB's web UI adapter framework
(`tvb/adapters/creators/allen_creator.py`), making it inaccessible without a
full TVB installation.  This project removes that dependency entirely —
exposing the same pipeline as a simple CLI that outputs plain CSV and NumPy
files.

## Origin & motivation

The connectivity builder was originally developed by Francesca Melozzi and
Marmaduke Woodman for The Virtual Brain.  It downloads tracer experiment
data from the Allen Institute, cleans and thresholds the projection maps,
and constructs a weighted structural connectivity matrix with region centres
and tract lengths.

For researchers who want the mouse connectivity data **without** installing
the entire TVB stack, this tool provides:

- A standalone installation with minimal dependencies
- A familiar `argparse` CLI with all parameters exposed
- CSV output that any analysis tool can read
- Disk caching so that subsequent runs reuse downloaded data

## Installation

```bash
pip install git+https://github.com/maedoc/allen-mouse-connectivity.git
```

Or, from a local clone:

```bash
git clone https://github.com/maedoc/allen-mouse-connectivity.git
cd allen-mouse-connectivity
pip install -e .
```

Requirements: Python ≥ 3.9 (use 3.11 for the smoothest experience) and a
network connection for the first data download.

## Quick start

### Verify the installation

```bash
# Check that the package imports and the Allen API is reachable (seconds)
python examples/demo.py --check
```

### Build connectivity (CLI)

```bash
# Default parameters (100 µm, PD/ID weighting, 80 % injection fraction)
# First run downloads ~3 GB from the Allen Institute; subsequent runs use cache.
allen-mouse-connectivity --output-dir ./my_connectivity
```

### Build connectivity (Python)

```python
from allen_mouse_connectivity import build_connectivity

results = build_connectivity(
    resolution=100,        # 100 µm (fastest; 25/50 for higher detail)
    weighting=1,           # 1 = PD/ID, 2 = PD, 3 = energy
    inj_f_thresh=0.8,      # 80% injection fraction threshold
    vol_thresh=1e9,         # minimum region volume
    cache_dir="./my_cache", # cache location (default: ~/.allen_mouse_cache)
)

print(results["weights"].shape)     # (2N, 2N)
print(results["region_labels"])     # ['Right ...', 'Left ...', ...]
print(results["n_regions"])         # number of regions (2N)
```

### Run the full demo

```bash
# Full pipeline: download data, build connectivity, validate, write output
python examples/demo.py --full
```

## Output files

All files are written to `--output-dir`:

| File | Shape | Description |
|---|---|---|
| `weights.csv` | (2N × 2N) | Structural connectivity matrix |
| `tract_lengths.csv` | (2N × 2N) | Euclidean tract lengths |
| `centres.csv` | (2N × 4) | `label,x,y,z` region centre coordinates |
| `region_labels.csv` | (2N) | Human-readable region names |
| `region_ids.csv` | (2N × 2) | `index,label` mapping |
| `parcellation.npy` | 3-D | Region-indexed parcellation volume |
| `template.npy` | 3-D | Allen template brain |
| `metadata.txt` | — | Resolution, shapes, region count |

Use `--volume-format csv` to write 3-D volumes as gzipped CSV (`x,y,z,value`
columns) instead of NumPy `.npy`.

## CLI reference

```
allen-mouse-connectivity \
  -o, --output-dir DIR         Output directory (default: .)
  --volume-format {npy,csv}    Format for 3-D volumes (default: npy)

  -r, --resolution {25,50,100} Spatial resolution in µm (default: 100)
  -w, --weighting WEIGHTING    pd_id | pd | energy (default: pd_id)
  --inj-f-thresh FLOAT         Min injected fraction 0.0–1.0 (default: 0.8)
  --vol-thresh FLOAT           Min region volume in µm³ (default: 1e9)
  --transgenic-line STR        Filter by Cre line, e.g. Emx1-IRES-Cre

  --cache-dir DIR              Allen SDK cache (default: ~/.allen_mouse_cache)
  --manifest-file FILE         Specific manifest file (overrides --cache-dir)

  -q, --quiet                  Suppress progress output
  -v, --verbose                Show debug logs
```

### Weighting schemes

| Option | Description |
|---|---|
| `pd_id` | Projection density divided by injection density (default) |
| `pd` | Raw projection density |
| `energy` | Projection energy |

## Caching

The Allen SDK's `MouseConnectivityCache` handles caching automatically.
Set `--cache-dir` to control where downloaded experiments and volumes are
stored.  The default is `~/.allen_mouse_cache`.

First run at 100 µm downloads approximately 3 GB of projection density and
annotation data.  25 µm and 50 µm resolutions require significantly more
disk space.  All subsequent runs reuse the cached data and complete in
seconds.

## Testing

```bash
# Run the smoke test suite (includes minimal API connectivity check)
python tests/test_smoke.py
```

The test suite verifies imports, CLI argument parsing, output writing, core
pipeline functions (`pms_cleaner`, `rotate_reference`, `construct_structural_conn`),
and a lightweight Allen API call.  It does **not** download full experiment
data, making it suitable for CI.

## Programmatic use — full reference

```python
from allen_mouse_connectivity import build_connectivity

results = build_connectivity(
    resolution=100,           # 25, 50, or 100 µm
    weighting=1,              # 1 = PD/ID, 2 = PD, 3 = energy
    inj_f_thresh=0.8,         # min injection fraction (0.0–1.0)
    vol_thresh=1e9,            # min region volume in µm³
    cache_dir="./my_cache",    # cache directory
    manifest_file=None,        # explicit manifest (overrides cache_dir)
    transgenic_line=False,     # e.g. 'Emx1-IRES-Cre' or False for all
    progress_callback=None,    # callable(stage_name, info_dict)
)

# Results dictionary keys:
#   weights         — (2N, 2N) normalised connectivity matrix
#   tract_lengths   — (2N, 2N) Euclidean tract lengths
#   centres         — (2N, 3) region centre coordinates
#   region_labels   — (2N,) region names ['Right ...', 'Left ...', ...]
#   vol_parcel      — 3D integer array (parcellation volume)
#   template        — 3D array (Allen template brain)
#   resolution      — spatial resolution used
#   n_regions       — total number of regions (2N = right + left)
```

## License

GPL-3.0-or-later.  Derived from The Virtual Brain.  See the
[original project](https://github.com/the-virtual-brain/tvb-mouse) for full
copyright details and citation instructions.