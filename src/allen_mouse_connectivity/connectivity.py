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
    ist2e = dictionary_builder(cache, transgenic_line)

    # 2 — Download projection matrices -----------------------------------------
    _report(progress_callback, "Downloading projection matrices",
            {"injection_sites": len(ist2e)})
    projmaps = download_an_construct_matrix(cache, weighting, ist2e,
                                            transgenic_line)

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
    """Build a dict mapping injection-structure-id → list of experiment ids."""
    all_experiments = mcc.get_experiments(dataframe=True, cre=transgenic_line)
    ist2e = {}
    for eid in all_experiments.index:
        isti = all_experiments.loc[eid]['primary_injection_structure']
        if isti not in ist2e:
            ist2e[isti] = []
        ist2e[isti].append(eid)
    return ist2e


def download_an_construct_matrix(mcc, weighting, ist2e, transgenic_line):
    """Download projection matrices and (optionally) injection densities."""
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
            all_experiments = mcc.get_experiments(dataframe=True,
                                                  cre=transgenic_line)
            for exp_id in all_experiments['id']:
                inj_d = mcc.get_injection_density(exp_id, file_name=None)
                injdensity[exp_id] = (
                    np.sum(inj_d[0]) / np.count_nonzero(inj_d[0]))
                logger.info("Experiment id %s, total injection density %s",
                            exp_id, injdensity[exp_id])
            for inj_id in range(len(list(projmaps.values()))):
                index = 0
                for exp_id in list(projmaps.values())[inj_id]['rows']:
                    list(projmaps.values())[inj_id]['matrix'][index] /= \
                        injdensity[exp_id]
                    index += 1
    return projmaps


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
    if len(sis0) != len(list(projmaps)):
        for inj_id in range(len(list(projmaps.values()))):
            targ_id = -1
            while len(list(projmaps.values())[inj_id]['columns']) != \
                    (3 * len(list(projmaps))):
                targ_id += 1
                col_struct = list(projmaps.values())[inj_id]['columns'][targ_id]
                if col_struct['structure_id'] not in list(projmaps):
                    del list(projmaps.values())[inj_id]['columns'][targ_id]
                    list(projmaps.values())[inj_id]['matrix'] = np.delete(
                        list(projmaps.values())[inj_id]['matrix'],
                        targ_id, 1)
                    targ_id = -1

    # 4 — Remove NaN-only areas iteratively ------------------------------------
    nan_id = {}
    for inj_id in projmaps:
        mat = projmaps[inj_id]['matrix']
        for targ_id in range(mat.shape[1]):
            if all(np.isnan(mat[exp, targ_id])
                   for exp in range(mat.shape[0])):
                if inj_id not in nan_id:
                    nan_id[inj_id] = []
                nan_id[inj_id].append(
                    projmaps[inj_id]['columns'][targ_id]['structure_id'])

    while bool(nan_id):
        remove = []
        nan_inj_max = 0
        # Find injection site with most NaN targets
        while list(nan_id)[0] != nan_inj_max:
            len_max = 0
            for inj_id in list(nan_id):
                if len(nan_id[inj_id]) > len_max:
                    nan_inj_max = inj_id
                    len_max = len(nan_id[inj_id])
            if list(nan_id)[0] != nan_inj_max:
                nan_id.pop(nan_inj_max)
                remove.append(nan_inj_max)
        if len(remove) == 0:
            for inj_id in nan_id:
                for target_id in nan_id[inj_id]:
                    if target_id not in remove:
                        remove.append(target_id)
        for rem in remove:
            if rem in list(projmaps):
                projmaps.pop(rem)
            for inj_id in range(len(list(projmaps))):
                targ_id = -1
                previous_size = len(list(projmaps.values())[inj_id]['columns'])
                while len(list(projmaps.values())[inj_id]['columns']) != \
                        (previous_size - 3):
                    targ_id += 1
                    column = list(projmaps.values())[inj_id]['columns'][targ_id]
                    if column['structure_id'] == rem:
                        del list(projmaps.values())[inj_id]['columns'][targ_id]
                        list(projmaps.values())[inj_id]['matrix'] = np.delete(
                            list(projmaps.values())[inj_id]['matrix'],
                            targ_id, 1)
                        targ_id = -1
        # Re-evaluate NaN
        nan_id = {}
        for inj_id in projmaps:
            mat = projmaps[inj_id]['matrix']
            for targ_id in range(mat.shape[1]):
                if all(np.isnan(mat[exp, targ_id])
                       for exp in range(mat.shape[0])):
                    if inj_id not in nan_id:
                        nan_id[inj_id] = []
                    nan_id[inj_id].append(
                        projmaps[inj_id]['columns'][targ_id]['structure_id'])

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
    for inj_id in range(len(list(projmaps.values()))):
        targ_id = -1
        while len(list(projmaps.values())[inj_id]['columns']) != \
                (len(id_ok) * 3):
            targ_id += 1
            col_struct = list(projmaps.values())[inj_id]['columns'][targ_id]
            if col_struct['structure_id'] not in id_ok:
                del list(projmaps.values())[inj_id]['columns'][targ_id]
                list(projmaps.values())[inj_id]['matrix'] = np.delete(
                    list(projmaps.values())[inj_id]['matrix'], targ_id, 1)
                targ_id = -1
    return projmaps


def infected_threshold(mcc, projmaps, inj_f_threshold):
    """Exclude experiments whose injected fraction is below *inj_f_threshold*."""
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
        if len(exp_not_accepted) < len(projmaps[ID]['rows']):
            id_ok.append(ID)
            projmaps[ID]['rows'] = list(
                set(projmaps[ID]['rows']).difference(set(exp_not_accepted)))
    for checkid in list(projmaps):
        if checkid not in id_ok:
            projmaps.pop(checkid, None)
    # Remove from target list (columns + matrix)
    for indexinj in range(len(list(projmaps.values()))):
        indextarg = -1
        while len(list(projmaps.values())[indexinj]['columns']) != \
                (len(id_ok) * 3):
            indextarg += 1
            col_struct = list(projmaps.values()
                              )[indexinj]['columns'][indextarg]
            if col_struct['structure_id'] not in id_ok:
                del list(projmaps.values())[indexinj]['columns'][indextarg]
                list(projmaps.values())[indexinj]['matrix'] = np.delete(
                    list(projmaps.values())[indexinj]['matrix'],
                    indextarg, 1)
                indextarg = -1
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
            matrix_temp = np.zeros((matrix.shape[1], 1), dtype=float)
            for i in range(matrix.shape[1]):
                if np.isnan(sum(matrix[:, i])):
                    occ = 0
                    for jj in range(matrix.shape[0]):
                        if matrix[jj, i] == matrix[jj, i]:  # NaN != NaN
                            occ += 1
                            matrix_temp[i, 0] += matrix[jj, i]
                    matrix_temp[i, 0] /= occ if occ else 1
                else:
                    matrix_temp[i, 0] = sum(matrix[:, i]) / matrix.shape[0]
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
    structural_conn /= np.amax(structural_conn)     # normalise
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
        if node_id in vol_r:
            vol_r[vol_r == node_id] = indexed_vec[index_vec]
            vol_l[vol_l == node_id] = indexed_vec[index_vec + left]
        children = structure_tree.children([node_id])[0]
        child_ids = [c['id'] for c in children]
        while child_ids:
            cid = child_ids.pop(0)
            if (cid in vol_r) and (cid not in projmaps):
                vol_r[vol_r == cid] = indexed_vec[index_vec]
                vol_l[vol_l == cid] = indexed_vec[index_vec + left]
        index_vec += 1

    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)

    # Assign unassigned voxels via parent ---------------------------------------
    bool_idx = vol_parcel > np.amax(indexed_vec)
    not_assigned = np.unique(vol_parcel[bool_idx])
    vol_r = vol_parcel[:, :, :(vol.shape[2] // 2)].astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] // 2):].astype(np.float64)

    for node_id in not_assigned:
        node_id = int(node_id)
        st = structure_tree.get_structures_by_id([node_id])[0]
        if st is not None:
            ancestor = list(st['structure_id_path'])
        else:
            ancestor = []
        while ancestor:
            pp = ancestor[-1]
            if pp in unique_parents:
                vol_r[vol_r == node_id] = indexed_vec[unique_parents[pp]]
                vol_l[vol_l == node_id] = indexed_vec[unique_parents[pp] + left]
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
        node_id = int(node_id)
        st = structure_tree.get_structures_by_id([node_id])[0]
        if st is not None:
            ancestor = list(st['structure_id_path'])
        else:
            ancestor = []
        while ancestor:
            pp = ancestor[-1]
            if pp in unique_grandparents:
                vol_r[vol_r == node_id] = indexed_vec[unique_grandparents[pp]]
                vol_l[vol_l == node_id] = indexed_vec[unique_grandparents[pp] + left]
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
    """
    # First rotation: x1=x2, y1=z2, z1=y2
    vol_trans = np.zeros((allen.shape[0], allen.shape[2], allen.shape[1]),
                         dtype=int)
    for x in range(allen.shape[0]):
        vol_trans[x, :, :] = allen[x, :, :][::-1].transpose()

    # Second rotation: x1=z2, y1=x1, z1=y2
    allen_rotate = np.zeros((allen.shape[2], allen.shape[0], allen.shape[1]),
                            dtype=int)
    for y in range(allen.shape[1]):
        allen_rotate[:, :, y] = vol_trans[:, :, y].transpose()

    return allen_rotate
