## ADDED Requirements

### Requirement: Vectorized rotate_reference
The `rotate_reference` function SHALL use `np.transpose` and slice operations instead of Python-level per-slice loops. The implementation SHALL be equivalent to rotating the Allen reference volume from (x1, y1, z1) to the TVB reference (x2, y2, z2) where x1=z2, y1=x2, z1=y2.

#### Scenario: Rotation preserves sum and shape
- **WHEN** a 3D array is passed to `rotate_reference`
- **THEN** the result SHALL have shape `(z_dim, x_dim, y_dim)` where the input shape was `(x_dim, y_dim, z_dim)`, and the sum of all elements SHALL be preserved

#### Scenario: Vectorized matches loop output
- **WHEN** the old loop-based and new vectorized implementations process the same input
- **THEN** they SHALL produce identical output arrays

### Requirement: Cache dict views in inner loops
Functions that iterate over `projmaps` and repeatedly access `list(projmaps.values())` or `list(projmaps.keys())` SHALL cache these lists before entering inner loops. Modifications to `projmaps` during iteration SHALL use the cached snapshot or rebuild the cache after mutations.

#### Scenario: Mutable dict iteration
- **WHEN** `pms_cleaner`, `areas_volume_threshold`, or `infected_threshold` modifies `projmaps` (deleting keys or columns)
- **THEN** the iteration SHALL work correctly against the pre-snapshot state, rebuilding the snapshot as needed after mutations