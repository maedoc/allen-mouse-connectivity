## Why

A code review revealed 4 critical bugs and 5 warnings in `connectivity.py` that cause silent data corruption in the scientific output. The tool can produce incorrect connectivity matrices, misassigned parcellation volumes, and crash on edge-case data. These must be fixed before the tool can be trusted for research use.

## What Changes

- **Fix division-by-zero** in `download_an_construct_matrix` when injection density volume is all zeros (produces inf/nan that silently corrupts the weight matrix)
- **Fix matrix-row desync** in `infected_threshold` — experiments removed from `rows` list but not from `matrix`, so excluded experiments still contribute to connectivity averages
- **Fix infinite-loop risk** in `pms_cleaner` NaN-cleanup step 4 — the `while list(nan_id)[0] != nan_inj_max` loop can fail to terminate
- **Fix float64==int comparison** in `mouse_brain_visualizer` — parcellation voxels compared to integer structure IDs after float64 conversion, causing silent mapping errors
- **Guard against zero-max normalization** in `construct_structural_conn` — divides by `np.amax(structural_conn)` which is zero when all connections are filtered out
- **Treat all-NaN columns as NaN (not zero)** in `construct_structural_conn` — `occ == 0` case yields 0 instead of NaN, hiding missing data as "no connection"
- **Reuse experiment list** in `download_an_construct_matrix` — the second `get_experiments()` call may return a different ordering, misaligning injection densities
- **Optimize `rotate_reference`** — replace Python slice loops with `np.transpose` for ~100× speedup
- **Optimize repeated `list(dict.values())`** — called inside inner loops, causing quadratic+ performance
- **Add targeted unit tests** for each bug fix to prevent regression

## Capabilities

### New Capabilities
- `bugfix-pipeline`: Fixes for the 4 critical bugs (division-by-zero, matrix-row desync, infinite loop, float comparison) and 2 warnings (zero-max normalization, all-NaN-as-zero)
- `pipeline-performance`: Performance optimizations for rotate_reference and repeated dict materialization
- `bugfix-regression-tests`: Unit tests covering each bug scenario with mocked Allen API responses

### Modified Capabilities
<!-- No existing specs -->

## Impact

- `src/allen_mouse_connectivity/connectivity.py` — all fixes modify core pipeline functions
- `tests/test_smoke.py` — new regression tests added
- No breaking API changes — all fixes preserve the public interface of `build_connectivity()`, `main()`, and CLI arguments
- Scientific output changes — connectivity matrices and parcellation volumes may differ from previous (buggy) runs, which is the intended correction