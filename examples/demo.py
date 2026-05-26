#!/usr/bin/env python3
"""Minimal working demo for allen-mouse-connectivity.

Demonstrates three usage patterns:
  1. CLI — one command to build a connectivity matrix
  2. Programmatic — Python API for full control
  3. Quick-check — verify results without re-downloading

Run this script directly:
    python examples/demo.py

The first run downloads Allen Institute data (~3 GB at 100 µm resolution).
Subsequent runs reuse the cached data.
"""

import os
import sys
import tempfile

# ── 1. CLI ──────────────────────────────────────────────────────────────────
#
#   The simplest way to build connectivity:
#
#       $ allen-mouse-connectivity --output-dir ./my_connectivity
#
#   This uses default settings (100 µm, PD/ID weighting, 80% injection fraction).
#   Output files are written to ./my_connectivity/:
#     - weights.csv            (2N × 2N structural connectivity matrix)
#     - tract_lengths.csv      (2N × 2N Euclidean tract lengths)
#     - centres.csv            (region centre coordinates)
#     - region_labels.csv      (human-readable region names)
#     - parcellation.npy       (3-D parcellation volume)
#     - template.npy           (3-D Allen template brain)
#     - metadata.txt           (resolution, shape, region count)
#
#   Common options:
#     -r 25|50|100             Resolution in µm (default: 100)
#     -w pd_id|pd|energy       Weighting scheme (default: pd_id)
#     --inj-f-thresh 0.8       Min injection fraction (default: 0.8)
#     --vol-thresh 1e9         Min region volume in µm³ (default: 1e9)
#     --cache-dir DIR          Cache directory (default: ~/.allen_mouse_cache)

# ── 2. Programmatic API ────────────────────────────────────────────────────

def demo_programmatic():
    """Build connectivity using the Python API and inspect the results."""
    import numpy as np
    from allen_mouse_connectivity import build_connectivity

    output_dir = os.path.join(tempfile.gettempdir(), "allen_demo_output")
    os.makedirs(output_dir, exist_ok=True)

    print("Building mouse connectivity from Allen Institute data...")
    print("(First run downloads ~3 GB; subsequent runs use cache.)\n")

    results = build_connectivity(
        resolution=100,           # 100 µm (fastest; 25/50 for higher detail)
        weighting=1,              # 1=PD/ID, 2=PD, 3=energy
        inj_f_thresh=0.8,         # 80% injection fraction threshold
        vol_thresh=1e9,           # minimum region volume
        cache_dir=os.path.join(tempfile.gettempdir(), "allen_demo_cache"),
    )

    # ── Inspect results ─────────────────────────────────────────────────
    w = results["weights"]
    tl = results["tract_lengths"]
    c = results["centres"]
    n = results["n_regions"]

    print(f"Regions:          {n}")
    print(f"Weights shape:    {w.shape}")
    print(f"Tract lengths:    {tl.shape}")
    print(f"Centres shape:    {c.shape}")
    print(f"Parcellation:     {results['vol_parcel'].shape}")
    print(f"Template:          {results['template'].shape}")
    print()
    print(f"Non-zero weights: {np.count_nonzero(w):,} / {w.size:,}")
    print(f"Max weight:       {w.max():.6f}")
    print(f"Min non-zero:     {w[w > 0].min():.6e}")
    print()
    print(f"First 5 regions:  {results['region_labels'][:5].tolist()}")

    # ── Sanity checks ──────────────────────────────────────────────────
    assert w.shape == (n, n), f"Shape mismatch: {w.shape} vs ({n}, {n})"
    assert tl.shape == (n, n), f"Shape mismatch: {tl.shape} vs ({n}, {n})"
    assert c.shape == (n, 3), f"Shape mismatch: {c.shape} vs ({n}, 3)"
    assert w.max() <= 1.0, f"Weights not normalised: max = {w.max()}"
    assert not np.any(np.isnan(w)), "Weights contain NaN!"
    assert not np.any(np.isinf(w)), "Weights contain inf!"
    print("\n✓ All sanity checks passed")

    # ── Save to disk ────────────────────────────────────────────────────
    from allen_mouse_connectivity.cli import _write_outputs
    _write_outputs(results, output_dir, "npy")
    print(f"\n✓ Output written to: {output_dir}")
    print("  Files:", "\n         ".join(sorted(os.listdir(output_dir))))

    return results


# ── 3. Quick-check ─────────────────────────────────────────────────────────

def demo_quick_check():
    """Verify the library imports and Allen API are reachable (no downloads)."""
    from allen_mouse_connectivity import build_connectivity, main

    assert build_connectivity is not None
    assert main is not None
    print("✓ Package imports correctly")

    # Verify Allen API connectivity (lightweight metadata call)
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MouseConnectivityCache(
                resolution=100,
                manifest_file=os.path.join(tmpdir, "manifest.json"),
            )
            experiments = cache.get_experiments(dataframe=True)
            assert len(experiments) > 0
            print(f"✓ Allen API reachable — {len(experiments)} experiments available")
    except ImportError:
        print("⊘ allensdk not installed — skipping API check")


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo_quick_check()
    elif "--full" in sys.argv:
        results = demo_programmatic()
    else:
        print(__doc__)
        print("\nUsage:")
        print("  python examples/demo.py --check   # Quick import + API check (seconds)")
        print("  python examples/demo.py --full    # Full pipeline (first run: ~10 min + download)")