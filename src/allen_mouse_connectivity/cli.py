# -*- coding: utf-8 -*-
"""
Command-line interface for building a mouse structural connectivity matrix
from Allen Institute tracer experiments.

Usage
-----
    python -m allen_mouse_connectivity_cli [OPTIONS]
    allen-mouse-connectivity [OPTIONS]

Examples
--------
    # Quick run with defaults (100 µm, PD/ID weighting, 80 % inj fraction)
    allen-mouse-connectivity --output-dir ./my_connectivity

    # High resolution, custom thresholds
    allen-mouse-connectivity --resolution 25 --vol-thresh 5e8 \\
        --inj-f-thresh 0.7 --output-dir ./results

    # Only download projection density, use a specific cache location
    allen-mouse-connectivity --weighting pd \\
        --cache-dir /data/allen_cache --output-dir ./results
"""

import argparse
import gzip
import logging
import os
import sys
import time

import numpy as np

from .connectivity import build_connectivity

logger = logging.getLogger(__name__)

RESOLUTION_CHOICES = (25, 50, 100)

WEIGHTING_CHOICES = {
    "pd_id": 1,
    "pd-id": 1,
    "1": 1,
    "pd": 2,
    "2": 2,
    "energy": 3,
    "3": 3,
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a mouse structural connectivity from the Allen "
                    "Institute Mouse Connectivity Atlas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Output ----------------------------------------------------------------
    out = parser.add_argument_group("Output options")
    out.add_argument(
        "-o", "--output-dir", default=".",
        help="Directory for output CSV/NPY files (default: current directory).",
    )
    out.add_argument(
        "--volume-format", choices=("npy", "csv"), default="npy",
        help="Format for 3-D volumes (parcellation and template). "
             "'npy' (default) is compact; 'csv' writes gzipped CSV with "
             "x,y,z,value columns.",
    )

    # --- Connectivity parameters -----------------------------------------------
    conn = parser.add_argument_group("Connectivity parameters")
    conn.add_argument(
        "-r", "--resolution", type=int, default=100,
        choices=RESOLUTION_CHOICES,
        help="Spatial resolution in microns (default: 100).",
    )
    conn.add_argument(
        "-w", "--weighting", default="pd_id",
        choices=list(WEIGHTING_CHOICES),
        help="Weighting scheme: pd_id (projection density / injection "
             "density, default), pd (projection density), "
             "energy (projection energy). "
             "Also accepts 1, 2, or 3.",
    )
    conn.add_argument(
        "--inj-f-thresh", type=float, default=0.8,
        help="Minimum injected fraction of voxels in the injection site "
             "(0.0–1.0, default: 0.8).",
    )
    conn.add_argument(
        "--vol-thresh", type=float, default=1e9,
        help="Minimum brain region volume in µm³ to be included "
             "(default: 1e9).",
    )
    conn.add_argument(
        "--transgenic-line", default=None,
        help="Filter experiments by transgenic Cre line (e.g. "
             "'Emx1-IRES-Cre').  Omit to include all experiments.",
    )

    # --- Cache -----------------------------------------------------------------
    cache = parser.add_argument_group("Cache options")
    cache.add_argument(
        "--cache-dir", default=None,
        help="Allen SDK cache directory "
             "(default: ~/.allen_mouse_cache).",
    )
    cache.add_argument(
        "--manifest-file", default=None,
        help="Path to a specific Allen SDK manifest file.  Overrides "
             "--cache-dir.",
    )

    # --- Logging ---------------------------------------------------------------
    log = parser.add_argument_group("Logging options")
    log.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages.",
    )
    log.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show debug-level logs.",
    )

    return parser.parse_args(argv)


def _setup_logging(quiet, verbose):
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CSV / NPY writers
# ---------------------------------------------------------------------------

def _write_csv(path, data, header=None, fmt="%.6e"):
    """Write a 2-D array as CSV."""
    np.savetxt(path, data, delimiter=",", header=header,
               fmt=fmt, comments="")


def _write_volume_csv(path, vol, origin=(0, 0, 0), voxel_size=(1, 1, 1)):
    """Write a 3-D volume as gzipped CSV with columns x, y, z, value."""
    xdim, ydim, zdim = vol.shape
    with gzip.open(path, "wt") as fh:
        fh.write("x,y,z,value\n")
        for x in range(xdim):
            for y in range(ydim):
                for z in range(zdim):
                    fh.write(f"{x},{y},{z},{vol[x, y, z]}\n")


def _write_outputs(results, output_dir, volume_format):
    """Persist all results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # 2-D matrices as CSV -------------------------------------------------------
    _write_csv(
        os.path.join(output_dir, "weights.csv"),
        results["weights"],
        header=",".join(results["region_labels"]),
    )
    logger.info("Wrote  weights.csv  (%s)", results["weights"].shape)

    _write_csv(
        os.path.join(output_dir, "tract_lengths.csv"),
        results["tract_lengths"],
        header=",".join(results["region_labels"]),
    )
    logger.info("Wrote  tract_lengths.csv  (%s)",
                results["tract_lengths"].shape)

    # Centres with label column -------------------------------------------------
    centres_with_labels = np.column_stack([
        results["region_labels"],
        results["centres"],
    ])
    _write_csv(
        os.path.join(output_dir, "centres.csv"),
        centres_with_labels,
        header="label,x,y,z",
        fmt="%s",
    )
    logger.info("Wrote  centres.csv  (%s)", results["centres"].shape)

    # Region labels -------------------------------------------------------------
    np.savetxt(
        os.path.join(output_dir, "region_labels.csv"),
        results["region_labels"],
        delimiter=",", fmt="%s", comments="",
        header="label",
    )
    logger.info("Wrote  region_labels.csv  (%d)", results["n_regions"])

    # Region ID mapping (index → label) -----------------------------------------
    ids = np.arange(results["n_regions"])
    id_label = np.column_stack([ids, results["region_labels"]])
    np.savetxt(
        os.path.join(output_dir, "region_ids.csv"),
        id_label,
        delimiter=",", fmt="%s", comments="",
        header="index,label",
    )
    logger.info("Wrote  region_ids.csv  (%d rows)", results["n_regions"])

    # 3-D volumes ---------------------------------------------------------------
    if volume_format == "npy":
        np.save(os.path.join(output_dir, "parcellation.npy"),
                results["vol_parcel"])
        logger.info("Wrote  parcellation.npy  (%s)",
                    results["vol_parcel"].shape)

        np.save(os.path.join(output_dir, "template.npy"),
                results["template"])
        logger.info("Wrote  template.npy  (%s)",
                    results["template"].shape)
    else:
        path = os.path.join(output_dir, "parcellation.csv.gz")
        _write_volume_csv(
            path, results["vol_parcel"],
            voxel_size=(results["resolution"],
                        results["resolution"],
                        results["resolution"]),
        )
        logger.info("Wrote  parcellation.csv.gz  (%s)",
                    results["vol_parcel"].shape)

        path = os.path.join(output_dir, "template.csv.gz")
        _write_volume_csv(path, results["template"])
        logger.info("Wrote  template.csv.gz  (%s)",
                    results["template"].shape)

    # Metadata ------------------------------------------------------------------
    meta_path = os.path.join(output_dir, "metadata.txt")
    with open(meta_path, "w") as fh:
        fh.write(f"resolution_um = {results['resolution']}\n")
        fh.write(f"n_regions     = {results['n_regions']}\n")
        fh.write(f"weights_shape = {results['weights'].shape}\n")
        fh.write(f"template_shape = {results['template'].shape}\n")
        fh.write(f"parcellation_shape = {results['vol_parcel'].shape}\n")
    logger.info("Wrote  metadata.txt")


# ---------------------------------------------------------------------------
# Progress callback for CLI
# ---------------------------------------------------------------------------

def _make_progress():
    """Return a progress callback that logs stages with elapsed time."""
    t0 = [time.time()]        # mutable cell so we can close over it

    def callback(stage, info):
        elapsed = time.time() - t0[0]
        extra = "  ".join(f"{k}={v}" for k, v in info.items())
        logger.info("[%5.1fs]  %s  %s", elapsed, stage, extra)

    return callback


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    _setup_logging(args.quiet, args.verbose)

    weighting = WEIGHTING_CHOICES[args.weighting]
    transgenic = (False if args.transgenic_line is None
                  else args.transgenic_line)

    logger.info("Allen Mouse Connectivity Builder")
    logger.info("  resolution    = %d µm", args.resolution)
    logger.info("  weighting     = %s (%d)", args.weighting, weighting)
    logger.info("  inj_f_thresh  = %.2f", args.inj_f_thresh)
    logger.info("  vol_thresh    = %.1e  µm³", args.vol_thresh)
    logger.info("  transgenic    = %s", transgenic)
    logger.info("  output_dir    = %s", args.output_dir)
    logger.info("  volume_format = %s", args.volume_format)
    logger.info("")

    results = build_connectivity(
        resolution=args.resolution,
        weighting=weighting,
        inj_f_thresh=args.inj_f_thresh,
        vol_thresh=args.vol_thresh,
        cache_dir=args.cache_dir,
        manifest_file=args.manifest_file,
        transgenic_line=transgenic,
        progress_callback=_make_progress(),
    )

    logger.info("")
    logger.info("Writing output files …")
    _write_outputs(results, args.output_dir, args.volume_format)

    logger.info("")
    logger.info("Done.  %d regions in connectivity.", results["n_regions"])
    logger.info("Output directory: %s",
                os.path.abspath(args.output_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
