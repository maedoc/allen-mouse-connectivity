## ADDED Requirements

### Requirement: Injection density division-by-zero guard
When computing `projection_density / injection_density` (weighting mode 1), the system SHALL check that `np.count_nonzero(inj_d[0]) > 0` before dividing. If an experiment's injection density volume is all zeros, the system SHALL skip that experiment (log a warning and exclude it from the injection density dictionary).

#### Scenario: Zero injection density
- **WHEN** an experiment has an injection density volume that is entirely zeros
- **THEN** the system SHALL log a warning and skip that experiment from the injection density map, and SHALL NOT produce inf or nan values in the connectivity matrix

#### Scenario: Normal injection density
- **WHEN** an experiment has a non-zero injection density
- **THEN** the system SHALL compute `np.sum(inj_d[0]) / np.count_nonzero(inj_d[0])` as before and include it normally

### Requirement: Matrix rows synchronized with experiment list
When `infected_threshold` removes experiments from `projmaps[ID]['rows']`, the system SHALL also remove the corresponding rows from `projmaps[ID]['matrix']`. The matrix rows SHALL be indexed by the `rows` list before removal.

#### Scenario: Some experiments filtered by injection fraction
- **WHEN** an injection site has 5 experiments and 2 fail the injection fraction threshold
- **THEN** the system SHALL remove those 2 experiments from both `rows` and `matrix`; the remaining 3 rows SHALL correspond to the 3 remaining experiment IDs

#### Scenario: All experiments at a site filtered
- **WHEN** all experiments at an injection site fail the injection fraction threshold
- **THEN** the entire injection site SHALL be removed from `projmaps`

### Requirement: NaN-cleanup loop terminates and produces correct result
The `pms_cleaner` step 4 (NaN-only area removal) SHALL use a straightforward find-remove-repeat algorithm that terminates. It SHALL NOT rely on `while list(nan_id)[0] != nan_inj_max` patterns that depend on dict insertion order.

#### Scenario: Structures with all-NaN columns
- **WHEN** a structure's column in the projection matrix contains all NaN values
- **THEN** the system SHALL remove that structure from both the injection site set and the target columns of all remaining projection maps, and SHALL repeat until no structures have all-NaN columns

#### Scenario: No NaN columns
- **WHEN** no structure has all-NaN columns after steps 1–3 of `pms_cleaner`
- **THEN** step 4 SHALL be a no-op and return `projmaps` unchanged

### Requirement: Float64-int comparison in parcellation
When comparing voxel values in `mouse_brain_visualizer` against structure IDs, the system SHALL cast the structure ID to the same dtype as the voxel array (`np.float64`) before comparison.

#### Scenario: Voxel matching with float64 volume
- **WHEN** `vol_r` and `vol_l` are converted to `float64` and compared against an integer `node_id`
- **THEN** the system SHALL use `np.float64(node_id)` for the comparison, ensuring the dtype match allows correct equality

### Requirement: Zero-max normalization guard
When computing `structural_conn /= np.amax(structural_conn)`, the system SHALL first check whether the maximum is zero. If the maximum is zero, the system SHALL leave the matrix as all-zeros (representing no connections).

#### Scenario: All connections filtered out
- **WHEN** aggressive thresholding removes all connections and `np.amax(structural_conn) == 0`
- **THEN** the system SHALL NOT divide by zero and SHALL produce an all-zeros weight matrix

#### Scenario: Normal connectivity matrix
- **WHEN** `np.amax(structural_conn) > 0`
- **THEN** the system SHALL normalize by the maximum as before

### Requirement: All-NaN columns produce NaN (not zero)
When `construct_structural_conn` encounters a column where all experiments have NaN values (`occ == 0`), the system SHALL set the averaged value to `np.nan` (not `0`).

#### Scenario: All-NaN column
- **WHEN** a target structure column has NaN for every experiment
- **THEN** the averaged value SHALL be `np.nan` to represent missing data

#### Scenario: Some valid values in column
- **WHEN** a target structure column has at least one non-NaN experiment value
- **THEN** the system SHALL average only the non-NaN values as before

### Requirement: Single experiment list for consistency
The `download_an_construct_matrix` function SHALL receive the experiment DataFrame from `dictionary_builder` rather than making a second `get_experiments()` call, ensuring consistency between the injection structure mapping and injection density calculations.

#### Scenario: Injection density uses same experiment set
- **WHEN** weighting mode 1 (PD/ID) is selected
- **THEN** the experiment IDs used for injection density SHALL come from the same DataFrame that built `ist2e`, not from a separate API call