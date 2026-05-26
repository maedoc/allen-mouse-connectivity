## Context

The `allen-mouse-connectivity` package downloads tracer experiment data from the Allen Institute Mouse Connectivity Atlas and builds a structural connectivity matrix. A code review identified 4 critical bugs and 5 warnings that cause silent data corruption, infinite-loop hangs, and performance degradation. All bugs are in `src/allen_mouse_connectivity/connectivity.py`. The CLI (`cli.py`) is unaffected except for the CSV volume writer performance.

The codebase is a standalone extraction from TVB's `allen_creator.py`. The bugs existed in the original code and were carried over during extraction.

## Goals / Non-Goals

**Goals:**
- Fix all 4 critical bugs that produce incorrect scientific output
- Fix 2 warnings that produce incorrect output in edge cases (zero-max normalization, all-NaN-as-zero)
- Fix 1 warning that may cause silent misalignment (independent `get_experiments()` call)
- Optimize `rotate_reference` with vectorized numpy (simple, high-impact)
- Reduce redundant `list(dict.values())` calls in inner loops
- Add regression tests for every fix using mocked Allen API data

**Non-Goals:**
- Refactoring the overall architecture or adding new features
- Changing the public API (`build_connectivity()`, CLI arguments, output format)
- Fixing the `_write_volume_csv` performance issue (separate concern, not a correctness bug)
- Adding type hints or full typing coverage
- Replacing the float-encoding scheme in `mouse_brain_visualizer` with integer remapping (too invasive for a bugfix release; the float==int comparison fix is sufficient)

## Decisions

### D1: Division-by-zero guard in `download_an_construct_matrix`
**Decision**: Add `np.count_nonzero(inj_d[0]) == 0` check; skip the experiment if injection density is all-zero, logging a warning.
**Rationale**: An all-zero injection density means the experiment has no usable data. Division produces inf/nan that propagates. Skip is the safest action.
**Alternative**: Set to 0 — would hide data absence. Set to NaN — would cascade through averaging. Skip is cleanest.

### D2: Sync matrix rows in `infected_threshold`
**Decision**: After removing experiments from `projmaps[ID]['rows']`, rebuild `projmaps[ID]['matrix']` keeping only rows matching the remaining experiment IDs.
**Rationale**: The bug is that `['rows']` is a list of experiment IDs but `['matrix']` is a positional numpy array. Removing from `rows` without removing the corresponding matrix rows means excluded experiments still contribute to connectivity averages.
**Alternative**: Build a boolean mask of kept rows and index the matrix. This is what we'll do — `keep_mask = [r in remaining_set for r in rows]`.

### D3: Rewrite NaN-cleanup loop in `pms_cleaner`
**Decision**: Replace the `while list(nan_id)[0] != nan_inj_max` loop with a straightforward approach: find which structure IDs have all-NaN columns, then remove them iteratively from both `projmaps` keys and column lists until stable.
**Rationale**: The original loop is convoluted and relies on dict ordering. A simpler find-remove-repeat loop is correct, readable, and terminates because each iteration removes at least one structure.

### D4: Fix float64==int comparison in `mouse_brain_visualizer`
**Decision**: Cast `node_id` to `float64` before comparison: `vol_r[vol_r == np.float64(node_id)]`.
**Rationale**: The Allen annotation volume voxels are integers, but after converting to float64 for the index-encoding scheme, `== int` comparisons may fail due to float representation. Casting the target avoids the mismatch.
**Alternative**: Use `np.isclose` — overkill for exact integers. Switch to integer parcellation — too invasive (rejected in non-goals).

### D5: Guard zero-max normalization
**Decision**: Check `np.amax(structural_conn) == 0` before dividing. If zero, leave the matrix as all-zeros (no connections).
**Rationale**: Division by zero produces nan/inf. Zero matrix is the correct semantic — no connections survived filtering.

### D6: All-NaN columns → NaN (not zero)
**Decision**: When `occ == 0` in `construct_structural_conn`, set the value to `np.nan` instead of `0`.
**Rationale**: Missing data should be NaN, not a fabricated zero. Downstream consumers can use `np.nanmean` or explicit NaN handling.

### D7: Reuse experiment list for injection density
**Decision**: Pass the experiment DataFrame from `dictionary_builder` into `download_an_construct_matrix` instead of making a second `get_experiments()` call.
**Rationale**: Two separate API calls may return different orderings or even different results if the database changes between calls. A single call guarantees consistency.
**Implementation**: Add `experiments_df` parameter to `download_an_construct_matrix`.

### D8: Vectorize `rotate_reference`
**Decision**: Replace slice loops with `np.transpose(allen, (2, 0, 1))[::-1].copy()`.
**Rationale**: The current implementation loops over slices in Python. The vectorized version is equivalent and ~100× faster for real volumes.
**Note**: The original code does two passes, but analytically the composition is `np.transpose(allen, (2,0,1))[::-1]` — flip the y-axis then transpose. Verified with the existing test.

### D9: Reduce `list(dict.values())` calls
**Decision**: Cache `list(projmaps.values())` and `list(projmaps.keys())` before inner loops.
**Rationale**: Each call materializes a new list. In inner loops with dict mutation via column deletion, this also causes incorrect indexing.

## Risks / Trade-offs

- **[Output will change]** — Fixes to bugs #1, #2, #5, #6 mean connectivity matrices will differ from previous (buggy) runs. This is intentional and correct, but users upgrading should be aware.
- **[D7 API change]** — Adding `experiments_df` parameter to `download_an_construct_matrix` changes its signature. This function is internal (not in `__init__.__all__`), so the risk is low.
- **[D3 rewrite]** — Rewriting the NaN-cleanup loop changes behavior on pathological inputs. The new logic is simpler and terminates, but may remove slightly different structures in rare edge cases. This is acceptable since the old logic could hang indefinitely.
- **[No full-pipeline integration test]** — A full end-to-end test requires downloading ~GB of data from the Allen API. We test with mocked data instead. This means subtle API-compatibility issues could still arise.