"""Quick smoke test for allen_mouse_connectivity.

Verifies:
1. Package imports cleanly (no TVB dependencies).
2. The Allen SDK MouseConnectivityCache can be instantiated.
3. The experiment metadata endpoint returns valid data.
4. CLI argument parsing and output writing work.

This test makes light API calls only — it does NOT download full experiment
data.  Suitable for CI.
"""

import os
import sys
import tempfile


def test_imports():
    """Core module imports without TVB."""
    from allen_mouse_connectivity.connectivity import (
        build_connectivity,
        dictionary_builder,
        pms_cleaner,
        rotate_reference,
    )
    from allen_mouse_connectivity.cli import main, _parse_args, _write_outputs
    assert build_connectivity is not None
    print("✓ imports OK")


def test_allensdk_api():
    """Minimal Allen SDK API call — metadata only, no volume downloads."""
    try:
        from allensdk.core.mouse_connectivity_cache import \
            MouseConnectivityCache
    except ImportError:
        print("⊘ allensdk not installed — skipping API test")
        return

    # Use a temp cache dir so we don't pollute the real cache
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "manifest.json")
        cache = MouseConnectivityCache(
            resolution=25,
            manifest_file=manifest,
        )

        # Fetch experiment metadata (lightweight HTTP call)
        experiments = cache.get_experiments(dataframe=True)

        assert experiments is not None
        assert len(experiments) > 0, "Expected at least one experiment"
        expected_cols = {"id", "primary_injection_structure"}
        missing = expected_cols - set(experiments.columns)
        assert not missing, f"Missing expected columns: {missing}"

        print(f"✓ Allen API returned {len(experiments)} experiments")


def test_cli_argparse():
    """Argparse parses all options with correct defaults."""
    from allen_mouse_connectivity.cli import _parse_args

    # Defaults
    args = _parse_args([])
    assert args.resolution == 100
    assert args.weighting == "pd_id"
    assert args.inj_f_thresh == 0.8
    assert args.vol_thresh == 1e9
    assert args.output_dir == "."
    assert args.volume_format == "npy"

    # Custom values
    args = _parse_args([
        "-o", "/tmp/out", "-r", "25", "-w", "pd",
        "--inj-f-thresh", "0.5", "--vol-thresh", "5e8",
        "--volume-format", "csv",
        "--transgenic-line", "Emx1-IRES-Cre",
        "--cache-dir", "/tmp/cache",
        "--manifest-file", "/tmp/manifest.json",
        "-v",
    ])
    assert args.output_dir == "/tmp/out"
    assert args.resolution == 25
    assert args.weighting == "pd"
    assert args.inj_f_thresh == 0.5
    assert args.vol_thresh == 5e8
    assert args.volume_format == "csv"
    assert args.transgenic_line == "Emx1-IRES-Cre"
    assert args.cache_dir == "/tmp/cache"
    assert args.manifest_file == "/tmp/manifest.json"
    assert args.verbose is True
    assert args.quiet is False

    print("✓ CLI argparse OK")


def test_output_writers():
    """CSV and NPY output writing with dummy data."""
    import numpy as np
    from allen_mouse_connectivity.cli import _write_outputs

    n = 4
    results = {
        "weights": np.eye(n, dtype=float),
        "tract_lengths": np.ones((n, n), dtype=float),
        "centres": np.random.default_rng(42).standard_normal((n, 3)),
        "region_labels": np.array(["R_a", "R_b", "L_a", "L_b"]),
        "vol_parcel": np.zeros((10, 10, 10), dtype=np.int32),
        "template": np.ones((10, 10, 10), dtype=float),
        "resolution": 100,
        "n_regions": n,
    }

    expected_csv = [
        "weights.csv", "tract_lengths.csv", "centres.csv",
        "region_labels.csv", "region_ids.csv", "metadata.txt",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_outputs(results, tmpdir, "csv")
        for fname in expected_csv + ["parcellation.csv.gz", "template.csv.gz"]:
            path = os.path.join(tmpdir, fname)
            assert os.path.exists(path), f"Missing: {fname}"
        print("✓ CSV output writers OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_outputs(results, tmpdir, "npy")
        for fname in expected_csv + ["parcellation.npy", "template.npy"]:
            path = os.path.join(tmpdir, fname)
            assert os.path.exists(path), f"Missing: {fname}"
        print("✓ NPY output writers OK")


def test_pms_cleaner_edge_cases():
    """pms_cleaner handles edge cases without crashing.

    Tests the logic with minimal projection map structures.  Does NOT hit the
    Allen API.
    """
    import numpy as np
    from allen_mouse_connectivity.connectivity import pms_cleaner

    # Minimal valid projection map — two injection sites, each with
    # columns for both sites × 3 hemispheres.
    projmaps = {
        502: {
            "columns": [
                {"structure_id": 502, "hemisphere_id": 2},
                {"structure_id": 502, "hemisphere_id": 1},
                {"structure_id": 502, "hemisphere_id": 3},
                {"structure_id": 1009, "hemisphere_id": 2},
                {"structure_id": 1009, "hemisphere_id": 1},
                {"structure_id": 1009, "hemisphere_id": 3},
            ],
            "matrix": np.ones((1, 6), dtype=float),
            "rows": [12345],
        },
        1009: {
            "columns": [
                {"structure_id": 502, "hemisphere_id": 2},
                {"structure_id": 502, "hemisphere_id": 1},
                {"structure_id": 502, "hemisphere_id": 3},
                {"structure_id": 1009, "hemisphere_id": 2},
                {"structure_id": 1009, "hemisphere_id": 1},
                {"structure_id": 1009, "hemisphere_id": 3},
            ],
            "matrix": np.ones((1, 6), dtype=float),
            "rows": [67890],
        },
    }

    result = pms_cleaner(projmaps)
    assert 502 in result
    assert 1009 in result

    print("✓ pms_cleaner edge cases OK")


def test_rotate_reference():
    """rotate_reference produces a permutation of the input."""
    import numpy as np
    from allen_mouse_connectivity.connectivity import rotate_reference

    arr = np.arange(24).reshape(2, 3, 4)
    rotated = rotate_reference(arr)
    assert rotated.shape == (4, 2, 3)  # known transformation
    assert rotated.sum() == arr.sum()
    print("✓ rotate_reference OK")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Smoke tests ===\n")
    failures = []

    for name, fn in [
        ("imports", test_imports),
        ("allensdk API", test_allensdk_api),
        ("CLI argparse", test_cli_argparse),
        ("output writers", test_output_writers),
        ("pms_cleaner edge cases", test_pms_cleaner_edge_cases),
        ("rotate_reference", test_rotate_reference),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failures.append(name)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All smoke tests passed.")
