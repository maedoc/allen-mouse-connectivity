## 1. Critical Bug Fixes

- [x] 1.1 Fix division-by-zero in `download_an_construct_matrix` — skip experiments with all-zero injection density, log warning (bugfix-pipeline spec, D1)
- [x] 1.2 Fix matrix-row desync in `infected_threshold` — rebuild matrix rows after filtering experiments from `rows` list (bugfix-pipeline spec, D2)
- [x] 1.3 Rewrite NaN-cleanup loop in `pms_cleaner` step 4 — replace `while list(nan_id)[0] != nan_inj_max` with a straightforward find-remove-repeat algorithm (bugfix-pipeline spec, D3)
- [x] 1.4 Fix float64==int comparison in `mouse_brain_visualizer` — cast `node_id` to `np.float64` before comparison (bugfix-pipeline spec, D4)
- [x] 1.5 Guard zero-max normalization in `construct_structural_conn` — check `np.amax == 0` before dividing (bugfix-pipeline spec, D5)
- [x] 1.6 Fix all-NaN columns producing 0 instead of NaN in `construct_structural_conn` (bugfix-pipeline spec, D6)
- [x] 1.7 Pass experiment DataFrame from `dictionary_builder` into `download_an_construct_matrix` to avoid second API call (bugfix-pipeline spec, D7)

## 2. Performance Optimizations

- [x] 2.1 Vectorize `rotate_reference` using `np.transpose` and `[::-1]` instead of Python loops (pipeline-performance spec, D8)
- [x] 2.2 Cache `list(projmaps.values())` and `list(projmaps.keys())` before inner loops in `pms_cleaner`, `areas_volume_threshold`, and `infected_threshold` (pipeline-performance spec, D9)

## 3. Regression Tests

- [x] 3.1 Add test for division-by-zero guard in injection density (mock experiment with all-zero density)
- [x] 3.2 Add test for matrix-row sync after `infected_threshold` filtering
- [x] 3.3 Add test for `pms_cleaner` NaN-cleanup terminating correctly
- [x] 3.4 Add test for float64-int voxel comparison in parcellation
- [x] 3.5 Add test for zero-max normalization guard
- [x] 3.6 Add test for all-NaN columns producing NaN (not zero)
- [x] 3.7 Verify all existing smoke tests still pass after changes