## ADDED Requirements

### Requirement: Regression tests for each bug fix
The test suite SHALL include unit tests that verify each bug fix with mocked or minimal data. Each test SHALL exercise the specific code path that was buggy and verify the corrected behavior.

#### Scenario: Division-by-zero guard
- **WHEN** an experiment has an all-zero injection density
- **THEN** the test SHALL verify that `download_an_construct_matrix` skips it without producing inf or nan

#### Scenario: Matrix-row desync after filtering
- **WHEN** experiments are removed by `infected_threshold`
- **THEN** the test SHALL verify that `projmaps[ID]['matrix']` has the same number of rows as `projmaps[ID]['rows']`

#### Scenario: NaN-cleanup termination
- **WHEN** `pms_cleaner` receives projection maps with all-NaN columns
- **THEN** the test SHALL verify that `pms_cleaner` terminates and removes those structures

#### Scenario: Float64-int comparison
- **WHEN** `mouse_brain_visualizer` compares float64 voxel values against integer structure IDs
- **THEN** the test SHALL verify that the comparison uses `np.float64(node_id)` and correctly matches voxels

#### Scenario: Zero-max normalization
- **WHEN** the connectivity matrix is all zeros after filtering
- **THEN** the test SHALL verify that `construct_structural_conn` does not produce nan or inf in the output

### Requirement: Existing tests continue to pass
All existing smoke tests in `tests/test_smoke.py` SHALL continue to pass after the bug fixes. No existing test SHALL be modified except to add new test cases.

#### Scenario: Smoke test suite
- **WHEN** `python tests/test_smoke.py` is executed
- **THEN** all existing test functions (`test_imports`, `test_allensdk_api`, `test_cli_argparse`, `test_output_writers`, `test_pms_cleaner_edge_cases`, `test_rotate_reference`) SHALL pass unchanged