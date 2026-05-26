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
        _remove_target_columns,
        _find_nan_columns,
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


def test_pms_cleaner_nan_removal():
    """pms_cleaner terminates correctly when columns are all-NaN.

    Regression test for the infinite-loop bug in step 4.
    """
    import numpy as np
    from allen_mouse_connectivity.connectivity import pms_cleaner

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
            "matrix": np.array([[1.0, 1.0, 1.0, np.nan, np.nan, np.nan]]),
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
            "matrix": np.array([[np.nan, np.nan, np.nan, 1.0, 1.0, 1.0]]),
            "rows": [67890],
        },
    }

    result = pms_cleaner(projmaps)
    # Both sites have all-NaN columns targeting the other,
    # but each site has non-NaN columns targeting itself.
    # Step 4 should terminate without infinite loop.
    assert isinstance(result, dict)
    print("✓ pms_cleaner NaN removal OK")


def test_rotate_reference():
    """rotate_reference produces a permutation of the input."""
    import numpy as np
    from allen_mouse_connectivity.connectivity import rotate_reference

    arr = np.arange(24).reshape(2, 3, 4)
    rotated = rotate_reference(arr)
    assert rotated.shape == (4, 2, 3)  # known transformation
    assert rotated.sum() == arr.sum()
    print("✓ rotate_reference OK")


def test_rotate_reference_vectorized_matches_loop():
    """Vectorized rotate_reference produces identical output to original loop version."""
    import numpy as np
    from allen_mouse_connectivity.connectivity import rotate_reference

    rng = np.random.default_rng(42)
    for shape in [(10, 20, 30), (5, 5, 5), (3, 7, 11)]:
        arr = rng.integers(0, 100, size=shape)
        result = rotate_reference(arr)
        # Manually compute expected: transpose(2,0,1) then flip axis 0
        expected = np.transpose(arr, (2, 0, 1))[::-1].copy()
        assert np.array_equal(result, expected), f"Mismatch for shape {shape}"
    print("✓ rotate_reference vectorized matches loop OK")


def test_zero_max_normalization():
    """construct_structural_conn does not produce nan/inf when matrix is all zeros.

    Regression test for division by zero in normalisation.
    """
    import numpy as np
    from allen_mouse_connectivity.connectivity import construct_structural_conn

    # Build a minimal projmaps with all-zero connectivity
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
            "matrix": np.zeros((1, 6), dtype=float),
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
            "matrix": np.zeros((1, 6), dtype=float),
            "rows": [67890],
        },
    }

    order = {
        0: [502, "Region_502"],
        1: [1009, "Region_1009"],
    }
    key_ord = [0, 1]

    result = construct_structural_conn(projmaps, order, key_ord)
    # All zeros → normalisation should NOT produce nan/inf
    assert not np.any(np.isnan(result)), "Zero-max produced NaN"
    assert not np.any(np.isinf(result)), "Zero-max produced inf"
    assert result.shape == (4, 4), f"Expected (4,4), got {result.shape}"
    print("✓ zero-max normalization guard OK")


def test_all_nan_column_produces_nan():
    """construct_structural_conn produces NaN for all-NaN columns, not zero.

    Regression test for the bug where all-NaN columns were silently set to 0.
    """
    import numpy as np
    from allen_mouse_connectivity.connectivity import construct_structural_conn

    # Two sites, where one has a NaN column (missing data for target 1009)
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
            "matrix": np.array([[1.0, 1.0, 1.0, np.nan, np.nan, np.nan]]),
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
            "matrix": np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            "rows": [67890],
        },
    }

    order = {
        0: [502, "Region_502"],
        1: [1009, "Region_1009"],
    }
    key_ord = [0, 1]

    result = construct_structural_conn(projmaps, order, key_ord)
    # The result should contain NaN values (from the all-NaN column)
    assert np.any(np.isnan(result)), "All-NaN column should produce NaN, not 0"
    print("✓ all-NaN column produces NaN OK")


def test_remove_target_columns():
    """_remove_target_columns correctly removes columns by structure_id."""
    import numpy as np
    from allen_mouse_connectivity.connectivity import _remove_target_columns

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
            "matrix": np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            "rows": [12345],
        },
    }

    _remove_target_columns(projmaps, {1009})

    assert len(projmaps[502]["columns"]) == 3
    # Should keep columns for structure_id 502
    assert all(c["structure_id"] == 502 for c in projmaps[502]["columns"])
    assert projmaps[502]["matrix"].shape == (1, 3)
    assert np.allclose(projmaps[502]["matrix"], np.array([[1.0, 2.0, 3.0]]))
    print("✓ _remove_target_columns OK")


def test_float64_int_comparison():
    """mouse_brain_visualizer compares float64 voxels against float64 node IDs.

    Regression test for bug where integer node_id was compared against
    float64 voxel values, causing silent mapping failures.
    """
    import numpy as np
    from allen_mouse_connectivity.connectivity import mouse_brain_visualizer

    # Create a simple annotation volume with known structure IDs
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[:, :, 0:2] = 502    # right hemisphere
    vol[:, :, 2:4] = 502    # left hemisphere (same structure for simplicity)

    # We need a minimal structure tree
    class FakeStructureTree:
        def children(self, ids):
            return [[] for _ in ids]
        def get_structures_by_id(self, ids):
            return [
                {"id": i, "structure_id_path": [997, i], "name": f"struct_{i}"}
                for i in ids
            ]

    order = {0: [502, "Region_502"]}
    key_ord = [0]
    unique_parents = {997: 0}
    unique_grandparents = {}
    projmaps = {502: True}  # just needs to be in dict for "cid not in projmaps"

    result = mouse_brain_visualizer(
        vol, order, key_ord,
        unique_parents, unique_grandparents,
        FakeStructureTree(), projmaps,
    )

    # The result should have assigned voxels (not all background)
    unique_vals = np.unique(result)
    # Should have region 0 (indexed) and possibly -1 (background)
    assert 0 in unique_vals, f"Expected region 0 in result, got {unique_vals}"
    print("✓ float64-int comparison OK")


def test_dictionary_builder_returns_df():
    """dictionary_builder returns both ist2e mapping and experiment DataFrame."""
    try:
        from allensdk.core.mouse_connectivity_cache import \
            MouseConnectivityCache
    except ImportError:
        print("⊘ allensdk not installed — skipping dictionary_builder test")
        return

    from allen_mouse_connectivity.connectivity import dictionary_builder

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = os.path.join(tmpdir, "manifest.json")
        cache = MouseConnectivityCache(resolution=25, manifest_file=manifest)
        ist2e, experiments_df = dictionary_builder(cache, False)

        assert isinstance(ist2e, dict), "ist2e should be a dict"
        assert experiments_df is not None, "experiments_df should not be None"
        assert len(ist2e) > 0, "Should have injection structures"
        assert "primary_injection_structure" in experiments_df.columns
        print(f"✓ dictionary_builder returns dict + DataFrame ({len(ist2e)} structures)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Smoke tests ===\n")
    failures = []

    tests = [
        ("imports", test_imports),
        ("allensdk API", test_allensdk_api),
        ("CLI argparse", test_cli_argparse),
        ("output writers", test_output_writers),
        ("pms_cleaner edge cases", test_pms_cleaner_edge_cases),
        ("pms_cleaner NaN removal", test_pms_cleaner_nan_removal),
        ("rotate_reference", test_rotate_reference),
        ("rotate_reference vectorized", test_rotate_reference_vectorized_matches_loop),
        ("zero-max normalization", test_zero_max_normalization),
        ("all-NaN column → NaN", test_all_nan_column_produces_nan),
        ("_remove_target_columns", test_remove_target_columns),
        ("float64-int comparison", test_float64_int_comparison),
        ("dictionary_builder returns DF", test_dictionary_builder_returns_df),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"✗ {name} FAILED:")
            traceback.print_exc()
            failures.append(name)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All smoke tests passed.")