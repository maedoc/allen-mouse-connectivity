# -*- coding: utf-8 -*-
#
# Standalone Allen Mouse Connectivity Builder
#
# Based on the Allen connectivity creator from The Virtual Brain
# (tvb-framework/tvb/adapters/creators/allen_creator.py).
#
# Original authors:
#   Francesca Melozzi <france.melozzi@gmail.com>
#   Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#

"""
Core logic for building a mouse structural connectivity matrix from Allen
Institute tracer experiments.  This module has no TVB dependencies — only
numpy and allensdk are required.

Orchestration
-------------
Call :func:`build_connectivity` with the desired parameters to download data
from the Allen API, construct the structural connectivity matrix, compute
region centres and tract lengths, and build a parcellation volume.
"""

import logging
import os.path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------------

def build_connectivity(resolution=100,
                       weighting=1,
                       inj_f_thresh=0.8,
                       vol_thresh=1e9,
                       cache_dir=None,
                       manifest_file=None,
                       transgenic_line=False,
                       progress_callback=None):
    """Build a mouse structural connectivity from the Allen Institute dataset.

    Parameters
    ----------
    resolution : int
        Spatial resolution in microns. One of 25, 50, or 100 (default).
    weighting : int
        Weighting scheme:
        1 — projection_density / injection_density (default)
        2 — projection_density
        3 — projection_energy
    inj_f_thresh : float
        Minimum injected fraction of voxels in the injection site (0.0–1.0).
        Default 0.8 (80 %).
    vol_thresh : float
        Minimum volume in µm³ for a brain region to be included.
        Default 1e9.
    cache_dir : str or None
        Directory for the Allen SDK cache.  When *manifest_file* is also
        given, *manifest_file* takes precedence.
    manifest_file : str or None
        Path to a specific Allen SDK manifest file.
    transgenic_line : bool or str
        Filter experiments by transgenic line.  ``False`` (default) returns
        all experiments; a string (e.g. ``'Emx1-IRES-Cre'``) filters.
    progress_callback : callable or None
        Called as ``callback(stage_name, info_dict)`` at each pipeline stage.

    Returns
    -------
    dict
        ``weights`` – (2N, 2N) structural connectivity matrix.
        ``tract_lengths`` – (2N, 2N) Euclidean tract lengths.
        ``centres`` – (2N, 3) region centre coordinates.
        ``region_labels`` – (2N,) region name strings.
        ``vol_parcel`` – 3D integer array (parcellation volume).
        ``template`` – 3D float array (Allen template brain).
        ``resolution`` – spatial resolution used.
        ``n_regions`` – total number of regions (2N).
    """
    # Resolve cache / manifest -------------------------------------------------
    if manifest_file is None:
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.allen_mouse_cache")
        os.makedirs(cache_dir, exist_ok=True)
        manifest_file = os.path.join(cache_dir, "mouse_connectivity_manifest.json")

    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

    cache = MouseConnectivityCache(resolution=resolution,
                                   manifest_file=manifest_file)

    # 1 — Build experiment dictionary ------------------------------------------
    _report(progress_callback, "Building experiment dictionary")
    ist2e, experiments_df = dictionary_builder(cache, transgenic_line)

    # 2 — Download projection matrices -----------------------------------------
    _report(progress_callback, "Downloading projection matrices",
            {"injection_sites": len(ist2e)})
    projmaps = download_an_construct_matrix(cache, weighting, ist2e,
                                            transgenic_line,
                                            experiments_df=experiments_df)

    # 3 — Clean projection maps ------------------------------------------------
    _report(progress_callback, "Cleaning projection maps",
            {"initial_sites": len(projmaps)})
    projmaps = pms_cleaner(projmaps)
    _report(progress_callback, "Projection maps cleaned",
            {"remaining_sites": len(projmaps)})

    # 4 — Get annotation & template volumes ------------------------------------
    _report(progress_callback, "Downloading annotation and template volumes")
    vol, _annot_info = cache.get_annotation_volume()
    template, _template_info = cache.get_template_volume()

    # 5 — Rotate template ------------------------------------------------------
    template = rotate_reference(template)

    # 6 — Get structure tree ---------------------------------------------------
    structure_tree = cache.get_structure_tree()

    # 7 — Volume threshold filter ----------------------------------------------
    _report(progress_callback, "Applying volume threshold",
            {"threshold_um3": vol_thresh})
    projmaps = areas_volume_threshold(cache, projmaps, vol_thresh, resolution)

    # 8 — Injection fraction threshold filter ----------------------------------
    _report(progress_callback, "Applying injection fraction threshold",
            {"threshold": inj_f_thresh})
    projmaps = infected_threshold(cache, projmaps, inj_f_thresh)

    # 9 — Region ordering ------------------------------------------------------
    _report(progress_callback, "Creating region file order")
    order, key_ord = create_file_order(projmaps, structure_tree)

    # 10 — Build structural connectivity matrix --------------------------------
    _report(progress_callback, "Building structural connectivity matrix")
    structural_conn = construct_structural_conn(projmaps, order, key_ord)

    # 11 — Compute region centres ----------------------------------------------
    _report(progress_callback, "Computing region centres")
    centres, names = construct_centres(cache, order, key_ord)

    # 12 — Compute tract lengths -----------------------------------------------
    _report(progress_callback, "Computing tract lengths")
    tract_lengths = construct_tract_lengths(centres)

    # 13 — Parent / grandparent mapping ----------------------------------------
    _report(progress_callback, "Finding parent/grandparent mappings")
    unique_parents, unique_grandparents = parents_and_grandparents_finder(
        cache, order, key_ord, structure_tree)

    # 14 — Build parcellation volume -------------------------------------------
    _report(progress_callback, "Building parcellation volume")
    vol_parcel = mouse_brain_visualizer(vol, order, key_ord,
                                        unique_parents, unique_grandparents,
                                        structure_tree, projmaps)

    n_regions = len(names)

    _report(progress_callback, "Done", {"n_regions": n_regions})

    return {
        "weights": structural_conn,
        "tract_lengths": tract_lengths,
        "centres": centres,
        "region_labels": np.array(names),
        "vol_parcel": vol_parcel,
        "template": template,
        "resolution": resolution,
        "n_regions": n_regions,
    }


def _report(callback, stage, info=None):
    logger.info("Stage: %s%s", stage,
                "  " + str(info) if info else "")
    if callback:
        callback(stage, info or {})


# ---------------------------------------------------------------------------
# Core pipeline functions (extracted from allen_creator.py)
# ---------------------------------------------------------------------------

def dictionary_builder(mcc, transgenic_line):
    """Build a dict mapping injection-structure-id → list of experiment ids.

    Returns
    -------
    tuple[dict, DataFrame]
        The ``ist2e`` mapping and the experiment DataFrame (reused later for
        injection density lookups, avoiding a second API call).
    """
    all_experiments = mcc.get_experiments(dataframe=True, cre=transgenic_line)
    ist2e = {}
    for eid in all_experiments.index:
        isti = all_experiments.loc[eid]['primary_injection_structure']
        if isti not in ist2e:
            ist2e[isti] = []
        ist2e[isti].append(eid)
    return ist2e, all_experiments


def download_an_construct_matrix(mcc, weighting, ist2e, transgenic_line,
                                 experiments_df=None):
    """Download projection matrices and (optionally) injection densities.

    Parameters
    ----------
    mcc : MouseConnectivityCache
    weighting : int
        1 = PD/ID, 2 = PD, 3 = energy.
    ist2e : dict
        Mapping from injection structure id → list of experiment ids.
    transgenic_line : bool or str
        Cre line filter.
    experiments_df : DataFrame or None
        If supplied (from :func:`dictionary_builder`), used for injection
        density lookups instead of a separate API call.  Avoids potential
        ordering mismatches between two independent ``get_experiments`` calls.
    """
    projmaps = {}
    if weighting == 3:                               # projection energy
        for isti, elist in ist2e.items():
            projmaps[isti] = mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),
                parameter='projection_energy')
            logger.info("injection site id %s has %d experiments with pm shape %s",
                        isti, len(elist), projmaps[isti]['matrix'].shape)
    else:                                             # projection density
        for isti, elist in ist2e.items():
            projmaps[isti] = mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),
                parameter='projection_density')
            logger.info("injection site id %s has %d experiments with pm shape %s",
                        isti, len(elist), projmaps[isti]['matrix'].shape)
        if weighting == 1:                            # PD / ID
            injdensity = {}
            # Use the DataFrame from dictionary_builder when available,
            # avoiding a second get_experiments() call that may return
            # a different ordering or set.
            if experiments_df is not None:
                all_experiments = experiments_df
            else:
                all_experiments = mcc.get_experiments(dataframe=True,
                                                      cre=transgenic_line)
            for exp_id in all_experiments['id']:
                inj_d = mcc.get_injection_density(exp_id, file_name=None)
                nonzero = np.count_nonzero(inj_d[0])
                if nonzero == 0:
                    logger.warning(
                        "Experiment %s has all-zero injection density; skipping",
                        exp_id)
                    continue
                injdensity[exp_id] = np.sum(inj_d[0]) / nonzero
                logger.info("Experiment id %s, total injection density %s",
                            exp_id, injdensity[exp_id])
            for isti in list(projmaps):
                pm = projmaps[isti]
                # Build a mask of rows whose experiment has a valid density
                kept = [i for i, eid in enumerate(pm['rows'])
                        if eid in injdensity]
                if not kept:
                    logger.warning(
                        "Injection site %s has no experiments with valid "
                        "injection density; removing site", isti)
                    del projmaps[isti]
                    continue
                # Divide kept rows by injection density
                for idx in kept:
                    eid = pm['rows'][idx]
                    pm['matrix'][idx] /= injdensity[eid]
    return projmaps


def _remove_target_columns(projmaps, remove_ids):
    """Remove columns whose structure_id is in *remove_ids* from every map.

    This is the shared logic for steps 3 of pms_cleaner and the column
    removal in areas_volume_threshold / infected_threshold.
    """
    remove_set = set(remove_ids)
    for inj_id in projmaps:
        pm = projmaps[inj_id]
        new_cols = []
        keep_mask = []
        for idx, col in enumerate(pm['columns']):
            if col['structure_id'] not in remove_set:
                new_cols.append(col)
                keep_mask.append(idx)
        pm['columns'] = new_cols
        if len(keep_mask) > 0:
            pm['matrix'] = pm['matrix'][:, keep_mask]
        else:
            pm['matrix'] = pm['matrix'][:, :0]


def _find_nan_columns(projmaps):
    """Return dict: inj_id → list of structure_ids whose column is all-NaN."""
    nan_id = {}
    for inj_id, pm in projmaps.items():
        mat = pm['matrix']
        # Find columns where every row is NaN
        all_nan_mask = np.all(np.isnan(mat), axis=0)
        for col_idx in np.where(all_nan_mask)[0]:
            sid = pm['columns'][col_idx]['structure_id']
            if inj_id not in nan_id:
                nan_id[inj_id] = []
            nan_id[inj_id].append(sid)
    return nan_id


def pms_cleaner(projmaps):
    """Clean projection maps in four steps.

    1. Remove injection sites whose target-set differs from the reference set.
    2. Remove injection sites that are not also target sites.
    3. Remove target sites that are not also injection sites.
    4. Iteratively remove structures that contain only NaN values.
    """
    def _get_structure_id_set(pm):
        return {c['structure_id'] for c in pm['columns']}

    sis0 = _get_structure_id_set(projmaps[502])

    # 1 — Uniform target sites -------------------------------------------------
    for inj_id in list(projmaps):
        sis_i = _get_structure_id_set(projmaps[inj_id])
        if len(sis0.difference(sis_i)) != 0:
            projmaps.pop(inj_id, None)

    # 2 — Injection sites must be targets --------------------------------------
    for inj_id in list(projmaps):
        if inj_id not in sis0:
            del projmaps[inj_id]

    # 3 — Target sites must be injection sites ---------------------------------
    injection_ids = set(projmaps.keys())
    target_ids = set()
    for pm in projmaps.values():
        for col in pm['columns']:
            target_ids.add(col['structure_id'])
    extra_targets = target_ids - injection_ids
    if extra_targets:
        _remove_target_columns(projmaps, extra_targets)

    # 4 — Remove NaN-only areas iteratively ------------------------------------
    # Find structure IDs whose column is all-NaN in every injection site that
    # contains them.  Remove those structures (and any injection sites that
    # are themselves all-NaN targets).  Repeat until stable.
    for _ in range(100):  # safety bound; typically converges in 2-3 rounds
        nan_id = _find_nan_columns(projmaps)
        if not nan_id:
            break
        # Collect all structure IDs that have at least one all-NaN column
        nan_structures = set()
        for sid_list in nan_id.values():
            nan_structures.update(sid_list)
        # Any injection site that IS a nan_structure gets removed entirely
        to_remove = nan_structures & set(projmaps.keys())
        # All nan_structures get removed from columns
        to_remove |= nan_structures
        if not to_remove:
            break
        # Remove injection sites
        for rid in list(to_remove & set(projmaps.keys())):
            projmaps.pop(rid)
        # Remove columns
        _remove_target_columns(projmaps, to_remove)

    return projmaps


def areas_volume_threshold(mcc, projmaps, vol_thresh, resolution):
    """Include only brain regions whose volume exceeds *vol_thresh* (µm³)."""
    threshold = vol_thresh / (resolution ** 3)
    id_ok = []
    for ID in projmaps:
        mask, _ = mcc.get_structure_mask(ID)
        tot_voxels = np.count_nonzero(mask) / 2   # mask has both hemispheres
        if tot_voxels > threshold:
            id_ok.append(ID)
    for checkid in list(projmaps):
        if checkid not in id_ok:
            projmaps.pop(checkid, None)
    # Remove from target list (columns + matrix)
    remove_ids = set(projmaps.keys()) - set(id_ok)
    # Actually: remove ids that are NOT in id_ok but ARE columns
    # We already removed keys from projmaps, need to remove their columns too
    col_remove = set()
    for pm in projmaps.values():
        for col in pm['columns']:
            if col['structure_id'] not in id_ok:
                col_remove.add(col['structure_id'])
    if col_remove:
        _remove_target_columns(projmaps, col_remove)
    return projmaps


def infected_threshold(mcc, projmaps, inj_f_threshold):
    """Exclude experiments whose injected fraction is below *inj_f_threshold*.

    After removing experiments from ``rows``, the corresponding rows in
    ``matrix`` are also removed so that the two stay synchronised.
    """
    id_ok = []
    for ID in projmaps:
        exp_not_accepted = []
        for exp in projmaps[ID]['rows']:
            inj_info = mcc.get_structure_unionizes(
                [exp], is_injection=True, structure_ids=[ID],
                include_descendants=True, hemisphere_ids=[2])
            if len(inj_info) == 0:
                exp_not_accepted.append(exp)
            else:
                inj_f = (inj_info['sum_projection_pixels'][0] /
                         inj_info['sum_pixels'][0])
                if inj_f < inj_f_threshold:
                    exp_not_accepted.append(exp)
        remaining = [e for e in projmaps[ID]['rows']
                     if e not in set(exp_not_accepted)]
        if len(remaining) < len(projmaps[ID]['rows']):
            # Synchronise matrix rows with the remaining experiment list
            pm = projmaps[ID]
            kept_set = set(remaining)
            keep_mask = [i for i, eid in enumerate(pm['rows'])
                         if eid in kept_set]
            pm['matrix'] = pm['matrix'][keep_mask]
            pm['rows'] = remaining
        if len(remaining) > 0:
            id_ok.append(ID)
    for checkid in list(projmaps):
        if checkid not in id_ok:
            projmaps.pop(checkid, None)
    # Remove from target list (columns + matrix)
    col_remove = set()
    for pm in projmaps.values():
        for col in pm['columns']:
            if col['structure_id'] not in id_ok:
                col_remove.add(col['structure_id'])
    if col_remove:
        _remove_target_columns(projmaps, col_remove)
    return projmaps


def create_file_order(projmaps, structure_tree):
    """Create graph-order-based region ordering and key list."""
    order = {}
    for index in range(len(projmaps)):
        target_id = list(projmaps.values())[0]['columns'][index]['structure_id']
        graph_order = structure_tree.get_structures_by_id(
            [target_id])[0]['graph_order']
        order[graph_order] = [
            target_id,
            structure_tree.get_structures_by_id([target_id])[0]['name'],
        ]
    key_ord = list(order)
    key_ord.sort()
    return order, key_ord


def construct_structural_conn(projmaps, order, key_ord):
    """Build the (2N, 2N) structural connectivity matrix."""
    len_right = len(list(projmaps))
    structural_conn = np.zeros((len_right, 2 * len_right), dtype=float)
    row = -1
    for graph_ord_inj in key_ord:
        row += 1
        inj_id = order[graph_ord_inj][0]
        target = projmaps[inj_id]['columns']
        matrix = projmaps[inj_id]['matrix']

        # Average over experiments, handling NaN
        if np.isnan(np.sum(matrix)):
            matrix_temp = np.full((matrix.shape[1], 1), np.nan, dtype=float)
            for i in range(matrix.shape[1]):
                non_nan = matrix[~np.isnan(matrix[:, i]), i]
                if len(non_nan) > 0:
                    matrix_temp[i, 0] = np.sum(non_nan) / matrix.shape[0]
                # else: stays NaN — missing data is NaN, not zero
            matrix = matrix_temp
        else:
            matrix = (np.array([sum(matrix[:, i])
                                for i in range(matrix.shape[1])]) /
                      matrix.shape[0])

        col = -1
        for graph_ord_targ in key_ord:
            col += 1
            targ_id = order[graph_ord_targ][0]
            for index in range(len(target)):
                if target[index]['structure_id'] == targ_id:
                    if target[index]['hemisphere_id'] == 2:
                        structural_conn[row, col] = matrix[index]
                    if target[index]['hemisphere_id'] == 1:
                        structural_conn[row, col + len_right] = matrix[index]

    # Mirror to both hemispheres
    first_quarter = structural_conn[:, :(structural_conn.shape[1] // 2)]
    second_quarter = structural_conn[:, (structural_conn.shape[1] // 2):]
    sc_down = np.concatenate((second_quarter, first_quarter), axis=1)
    structural_conn = np.concatenate((structural_conn, sc_down), axis=0)
    # Normalise — guard against all-zero matrix
    max_val = np.amax(structural_conn)
    if max_val > 0:
        structural_conn /= max_val
    return structural_conn.T


def construct_centres(mcc, order, key_ord):
    """Compute region centre coordinates and names."""
    centres = np.zeros((len(key_ord) * 2, 3), dtype=float)
    names = []
    row = -1
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        coord = [0.0, 0.0, 0.0]
        mask, _ = mcc.get_structure_mask(node_id)
        mask = rotate_reference(mask)
        mask_r = mask[:mask.shape[0] // 2, :, :]
        xyz = np.where(mask_r)
        if xyz[0].shape[0] > 0:
            coord[0] = np.mean(xyz[0])
            coord[1] = np.mean(xyz[1])
            coord[2] = np.mean(xyz[2])
        row += 1
        centres[row, :] = coord
        coord[0] = mask.shape[0] - coord[0]
        centres[row + len(key_ord), :] = coord
        names.append('Right ' + str(order[graph_ord_inj][1]))
    for graph_ord_inj in key_ord:
        names.append('Left ' + str(order[graph_ord_inj][1]))
    return centres, names


def construct_tract_lengths(centres):
    """Compute Euclidean tract lengths between region centres."""
    len_right = len(centres) // 2
    tracts = np.zeros((len_right, len(centres)), dtype=float)
    for inj in range(len_right):
        center_inj = centres[inj]
        for targ in range(len_right):
            targ_r = centres[targ]
            targ_l = centres[targ + len_right]
            tracts[inj, targ] = np.sqrt(
                (center_inj[0] - targ_r[0]) ** 2 +
                (center_inj[1] - targ_r[1]) ** 2 +
                (center_inj[2] - targ_r[2]) ** 2)
            tracts[inj, targ + len_right] = np.sqrt(
                (center_inj[0] - targ_l[0]) ** 2 +
                (center_inj[1] - targ_l[1]) ** 2 +
                (center_inj[2] - targ_l[2]) ** 2)
    # Mirror to both hemispheres
    first_quarter = tracts[:, :(tracts.shape[1] // 2)]
    second_quarter = tracts[:, (tracts.shape[1] // 2):]
    tracts_down = np.concatenate((second_quarter, first_quarter), axis=1)
    tracts = np.concatenate((tracts, tracts_down), axis=0)
    return tracts.T


def parents_and_grandparents_finder(mcc, order, key_ord, structure_tree):
    """Map parent / grandparent structure IDs to the largest child region."""
    parents = []
    grandparents = []
    vol_areas = []
    vec_index = []
    index = 0
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        sid_path = structure_tree.get_structures_by_id(
            [node_id])[0]['structure_id_path']
        parents.append(sid_path[-2])
        grandparents.append(sid_path[-3])
        vec_index.append(index)
        index += 1
        mask, _ = mcc.get_structure_mask(node_id)
        vol_areas.append(np.count_nonzero(mask))

    # Sort by volume (ascending) — last (largest) wins for each parent
    parents = [p for (_v, p) in sorted(zip(vol_areas, parents))]
    grandparents = [g for (_v, g) in sorted(zip(vol_areas, grandparents))]
    vec_index = [i for (_v, i) in sorted(zip(vol_areas, vec_index))]

    k = len(parents)
    unique_parents = {}
    for p in reversed(parents):
        k -= 1
        if p not in unique_parents:
            unique_parents[p] = vec_index[k]

    k = len(grandparents)
    unique_grandparents = {}
    for p in reversed(grandparents):
        k -= 1
        if not np.isnan(p) and p not in unique_grandparents:
            unique_grandparents[p] = vec_index[k]

    return unique_parents, unique_grandparents


def mouse_brain_visualizer(vol, order, key_ord,
                           unique_parents, unique_grandparents,
                           structure_tree, projmaps):
    """Create a parcellation volume indexed 0..N-1 (-1 = background)."""
    tot_areas = len(key_ord) * 2
    indexed_vec = np.arange(tot_areas, dtype=float).reshape(tot_areas)
    indexed_vec += 1
    indexed_vec *= 10 ** (-(1 + int(np.log10(tot_areas))))

    vol_r = vol[:, :, :(vol.shape[2] // 2)].astype(np.float64)
    vol_l = vol[:, :, (vol.shape[2] // 2):].astype(np.float64)
    left = len(indexed_vec) // 2

    index_vec = 0
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        # Fix: cast node_id to float64 for comparison with float64 voxel values
        node_id_f = np.float64(node_id)
        if node_id_f in vol_r:
            vol_r[vol_r == node_id_f] = indexed_vec[index_vec]
            vol_l[vol_l == node_id_f] = indexed_vec[index_vec + left]
        children = structure_tree.children([node_id])[0]
        child_ids = [c['id'] for c in children]
        while child_ids:
            cid = child_ids.pop(0)
            cid_f = np.float64(cid)
            if (cid_f in vol_r) and (cid not in projmaps):
                vol_r[vol_r == cid_f] = indexed_vec[index_vec]
                vol_l[vol_l == cid_f] = indexed_vec[index_vec + left]
        index_vec += 1

    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)

    # Assign unassigned voxels via parent ---------------------------------------
    bool_idx = vol_parcel > np.amax(indexed_vec)
    not_assigned = np.unique(vol_parcel[bool_idx])
    vol_r = vol_parcel[:, :, :(vol.shape[2] // 2)].astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] // 2):].astype(np.float64)

    for node_id in not_assigned:
        node_id_int = int(node_id)
        node_id_f = np.float64(node_id_int)
        st = structure_tree.get_structures_by_id([node_id_int])[0]
        if st is not None:
            ancestor = list(st['structure_id_path'])
        else:
            ancestor = []
        while ancestor:
            pp = ancestor[-1]
            if pp in unique_parents:
                vol_r[vol_r == node_id_f] = indexed_vec[unique_parents[pp]]
                vol_l[vol_l == node_id_f] = indexed_vec[unique_parents[pp] + left]
                ancestor = []
            else:
                ancestor.remove(pp)
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)

    # Assign unassigned voxels via grandparent ----------------------------------
    bool_idx = vol_parcel > np.amax(indexed_vec)
    not_assigned = np.unique(vol_parcel[bool_idx])
    vol_r = vol_parcel[:, :, :(vol.shape[2] // 2)].astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] // 2):].astype(np.float64)

    for node_id in not_assigned:
        node_id_int = int(node_id)
        node_id_f = np.float64(node_id_int)
        st = structure_tree.get_structures_by_id([node_id_int])[0]
        if st is not None:
            ancestor = list(st['structure_id_path'])
        else:
            ancestor = []
        while ancestor:
            pp = ancestor[-1]
            if pp in unique_grandparents:
                vol_r[vol_r == node_id_f] = indexed_vec[unique_grandparents[pp]]
                vol_l[vol_l == node_id_f] = indexed_vec[unique_grandparents[pp] + left]
                ancestor = []
            else:
                ancestor.remove(pp)
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)

    vol_parcel[vol_parcel >= 1] = 0
    vol_parcel *= 10 ** (1 + int(np.log10(tot_areas)))
    vol_parcel -= 1
    vol_parcel = np.round(vol_parcel)
    vol_parcel = rotate_reference(vol_parcel)
    return vol_parcel


def rotate_reference(allen):
    """Rotate Allen 3D reference (x1,y1,z1) → TVB reference (x2,y2,z2).

    Relationship: x1=z2, y1=x2, z1=y2.

    Equivalent to the original two-pass slice loop but expressed as a
    single vectorised :func:`numpy.transpose` + flip operation.
    """
    # Original: first rotate (x, z, y) with y-flip, then rotate (z, x, y).
    # The composition equals: transpose(2, 0, 1) with axis-0 reversed.
    return np.transpose(allen, (2, 0, 1))[::-1].copy()