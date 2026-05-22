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

Or, from PyPI (once published):

```bash
pip install allen-mouse-connectivity
```

If you prefer a local editable install for development:

```bash
git clone https://github.com/maedoc/allen-mouse-connectivity.git
cd allen-mouse-connectivity
pip install -e .
```

Requirements: Python ≥ 3.9 (use 3.11 for the smoothest experience) and a
network connection for the first data download.

## Quick start

```bash
# Default parameters (100 µm, PD/ID weighting, 80 % injection fraction)
allen-mouse-connectivity --output-dir ./my_connectivity
```

The first run downloads experimental data from the Allen Institute (typically
a few GB depending on resolution).  Subsequent runs are fast thanks to disk
caching.

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

## Programmatic use

```python
from allen_mouse_connectivity import build_connectivity

results = build_connectivity(
    resolution=100,
    weighting=1,          # PD / ID
    inj_f_thresh=0.8,
    vol_thresh=1e9,
    cache_dir="./my_cache",
)

print(results["weights"].shape)     # (2N, 2N)
print(results["region_labels"])     # ['Right ...', 'Left ...', ...]
```

## License

GPL-3.0-or-later.  Derived from The Virtual Brain.  See the
[original project](https://github.com/the-virtual-brain/tvb-mouse) for full
copyright details and citation instructions.
