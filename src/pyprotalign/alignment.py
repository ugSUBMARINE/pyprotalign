"""Sequence alignment operations."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .chain import ProteinChain
from .gemmi_utils import align_sequences
from .kabsch import calculate_rmsd, superpose
from .refine import iterative_superpose

logger = logging.getLogger(__name__)


def align_two_chains(
    fixed_chain: ProteinChain,
    mobile_chain: ProteinChain,
    refine: bool = False,
    cutoff_factor: float = 2.0,
    max_cycles: int = 5,
    filter: bool = False,
    min_bfactor: float = -np.inf,
    max_bfactor: float = np.inf,
    min_occ: float = -np.inf,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Align two protein chains based on a sequence alignment.
       Allows filtering by B-factor /pLDDT. Optionally refines alignment.

    Args:
        fixed_chain: Fixed ProteinChain
        mobile_chain: Mobile ProteinChain
        refine: Enable iterative refinement (optional)
        cutoff_factor: Outlier rejection cutoff for refinement
        max_cycles: Maximum refinement cycles
        filter: Whether to apply quality filtering based on B-factor/pLDDT and occupancy (optional)
        min_bfactor: Minimum B-factor threshold for quality filtering
        max_bfactor: Maximum B-factor threshold for quality filtering
        min_occ: Minimum occupancy threshold for quality filtering

    Returns:
        Tuple of (rotation, translation, rmsd, num_aligned):
        - rotation: 3x3 rotation matrix
        - translation: 3-element translation vector
        - rmsd: RMSD of aligned CA atoms
        - num_aligned: Number of aligned CA atom pairs

    Raises:
        ValueError: If fewer than 3 aligned CA pairs
    """
    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    # Align sequences
    pairs = align_sequences(fixed_chain.sequence, mobile_chain.sequence)

    # Extract aligned residue indices (where both have CA atoms)
    fixed_indices, mobile_indices = _get_indices_from_pairs(fixed_chain, mobile_chain, pairs)

    if len(fixed_indices) < 3:
        raise ValueError(f"Need at least 3 aligned CA pairs, found {len(fixed_indices)}")

    # Get coordinates at the aligned indices
    fixed_coords = fixed_chain.coords[fixed_indices]
    mobile_coords = mobile_chain.coords[mobile_indices]

    if debug_enabled:
        logger.debug("%d CA pairs after sequence alignment.", len(fixed_coords))

    # Apply quality filtering if requested
    if filter:
        fixed_mask = fixed_chain.get_bfac_occ_mask(min_bfactor, max_bfactor, min_occ, fixed_indices)
        mobile_mask = mobile_chain.get_bfac_occ_mask(min_bfactor, max_bfactor, min_occ, mobile_indices)
        quality_mask = fixed_mask & mobile_mask

        fixed_coords = fixed_coords[quality_mask]
        mobile_coords = mobile_coords[quality_mask]

        if len(fixed_coords) < 3:
            raise ValueError(f"Need at least 3 CA pairs after quality filtering, found {len(fixed_coords)}")

        if debug_enabled:
            logger.debug(
                "%d CA pairs after quality filtering (%.1f <= B <= %.1f, occ >= %.1f).",
                len(fixed_coords),
                min_bfactor,
                max_bfactor,
                min_occ,
            )

    # Compute transformation
    if refine:
        rotation, translation, mask, rmsd = iterative_superpose(
            fixed_coords, mobile_coords, max_cycles=max_cycles, cutoff_factor=cutoff_factor
        )
        num_aligned = int(np.sum(mask))
    else:
        rotation, translation = superpose(fixed_coords, mobile_coords)
        mobile_transformed = mobile_coords @ rotation.T + translation
        rmsd = calculate_rmsd(fixed_coords, mobile_transformed)
        num_aligned = len(fixed_coords)

    return rotation, translation, rmsd, num_aligned


def _get_indices_from_pairs(
    chn_1: ProteinChain, chn_2: ProteinChain, pairs: tuple[tuple[int | None, int | None], ...]
) -> tuple[list[int], list[int]]:
    """Helper to extract aligned residue indices from sequence alignment pairs.

    Args:
        chn_1: First ProteinChain
        chn_2: Second ProteinChain
        pairs: Tuple of (idx1, idx2) from align_sequences

    Returns:
        Tuple of (list of indices in chn_1, list of indices in chn_2)
    """
    chn_1_indices = []
    chn_2_indices = []
    for idx_1, idx_2 in pairs:
        # Skip gaps
        if idx_1 is None or idx_2 is None:
            continue
        # Check both residues have CA atoms (not NaN)
        if np.isnan(chn_1.coords[idx_1, 0]) or np.isnan(chn_2.coords[idx_2, 0]):
            continue

        chn_1_indices.append(idx_1)
        chn_2_indices.append(idx_2)

    return chn_1_indices, chn_2_indices


def align_globally(
    fixed_chain_map: dict[str, ProteinChain],
    mobile_chain_map: dict[str, ProteinChain],
    by_chain_id: bool = True,
    refine: bool = False,
    cutoff_factor: float = 2.0,
    max_cycles: int = 5,
    filter: bool = False,
    min_bfactor: float = -np.inf,
    max_bfactor: float = np.inf,
    min_occ: float = -np.inf,
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, str]]:
    """Align multiple chains.

    Matches chains either by matching chain IDs or by by their order in the provided maps,
    aligns sequences for each pair, and pools all aligned CA atom coordinates across chains.

    Args:
        fixed_chain_map: Mapping of chain ID to ProteinChain for fixed structure
        mobile_structure: Mapping of chain ID to ProteinChain for fixed structure
        by_chain_id: Match chains by ID (True) or by their order (False)
        refine: Enable iterative refinement (optional)
        cutoff_factor: Outlier rejection cutoff for refinement
        max_cycles: Maximum refinement cycles
        filter: Whether to apply quality filtering based on B-factor/pLDDT and occupancy (optional)
        min_bfactor: Minimum B-factor threshold for quality filtering
        max_bfactor: Maximum B-factor threshold for quality filtering
        min_occ: Minimum occupancy threshold for quality filtering

    Returns:
        Tuple of (rotation, translation, rmsd, num_aligned):
        - rotation: 3x3 rotation matrix
        - translation: 3-element translation vector
        - rmsd: RMSD of aligned CA atoms across all chains
        - num_aligned: Total number of aligned CA atom pairs across all chains
        - chain_mapping: Dictionary mapping fixed chain IDs to mobile chain IDs used for alignment

    Raises:
        ValueError: If no matching protein chains found or fewer than 3 aligned pairs
    """
    if by_chain_id:
        # Find matching chain names
        common_chains = set(fixed_chain_map.keys()) & set(mobile_chain_map.keys())

        if not common_chains:
            raise ValueError("No matching chain labels found between structures.")

        chain_mapping = {chain_id: chain_id for chain_id in sorted(common_chains)}
    else:
        # Match chains by their order in the provided maps
        # This allows aligning chains with different IDs but the same order (e.g. A-B-C vs D-E-F)
        chain_mapping = {
            chn_1.chain_id: chn_2.chain_id
            for chn_1, chn_2 in zip(fixed_chain_map.values(), mobile_chain_map.values(), strict=False)
        }

    rotation, translation, rmsd, num_aligned = align_mapped_chains(
        fixed_chain_map,
        mobile_chain_map,
        chain_mapping,
        refine,
        cutoff_factor,
        max_cycles,
        filter,
        min_bfactor,
        max_bfactor,
        min_occ,
    )

    return rotation, translation, rmsd, num_aligned, chain_mapping


def align_mapped_chains(
    fixed_chain_map: dict[str, ProteinChain],
    mobile_chain_map: dict[str, ProteinChain],
    chain_mapping: dict[str, str],
    refine: bool = False,
    cutoff_factor: float = 2.0,
    max_cycles: int = 5,
    filter: bool = False,
    min_bfactor: float = -np.inf,
    max_bfactor: float = np.inf,
    min_occ: float = -np.inf,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Align multiple chains based on mapped chain IDs.

    Matches mapped chains (defined in 'chain_mapping'), aligns sequences for each pair,
    and pools all aligned CA atom coordinates across chains.

    Args:
        fixed_chain_map: Mapping of chain ID to ProteinChain for fixed structure
        mobile_structure: Mapping of chain ID to ProteinChain for fixed structure
        chain_mapping: Mapping of fixed chain ID to mobile chain ID for alignment
        refine: Enable iterative refinement (optional)
        cutoff_factor: Outlier rejection cutoff for refinement
        max_cycles: Maximum refinement cycles
        filter: Whether to apply quality filtering based on B-factor/pLDDT and occupancy (optional)
        min_bfactor: Minimum B-factor threshold for quality filtering
        max_bfactor: Maximum B-factor threshold for quality filtering
        min_occ: Minimum occupancy threshold for quality filtering

    Returns:
        Tuple of (rotation, translation, rmsd, num_aligned):
        - rotation: 3x3 rotation matrix
        - translation: 3-element translation vector
        - rmsd: RMSD of aligned CA atoms across all chains
        - num_aligned: Total number of aligned CA atom pairs across all chains

    Raises:
        ValueError: If no matching protein chains found or fewer than 3 aligned pairs
    """
    # Collect coordinates across all matched chains
    fixed_coords_list = []
    mobile_coords_list = []

    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    for chain_id_fixed, chain_id_mobile in chain_mapping.items():
        fixed_chain = fixed_chain_map[chain_id_fixed]
        mobile_chain = mobile_chain_map[chain_id_mobile]

        if debug_enabled:
            logger.debug(
                "\n-- Chain mapping %s → %s --",
                chain_id_fixed,
                chain_id_mobile,
            )

        # Align sequences
        pairs = align_sequences(fixed_chain.sequence, mobile_chain.sequence)

        # Extract aligned residue indices (where both have CA atoms)
        fixed_indices, mobile_indices = _get_indices_from_pairs(fixed_chain, mobile_chain, pairs)

        if len(fixed_indices) == 0:
            logger.warning(
                "Mapping %s → %s: No aligned CA pairs. Skipping this pair of chains.",
                chain_id_fixed,
                chain_id_mobile,
            )
            continue

        # Get coordinates at the aligned indices
        fixed_coords = fixed_chain.coords[fixed_indices]
        mobile_coords = mobile_chain.coords[mobile_indices]

        if debug_enabled:
            logger.debug("%d CA pairs after sequence alignment.", len(fixed_coords))

        # Apply quality filtering if requested
        if filter:
            fixed_mask = fixed_chain.get_bfac_occ_mask(min_bfactor, max_bfactor, min_occ, fixed_indices)
            mobile_mask = mobile_chain.get_bfac_occ_mask(min_bfactor, max_bfactor, min_occ, mobile_indices)
            quality_mask = fixed_mask & mobile_mask

            fixed_coords = fixed_coords[quality_mask]
            mobile_coords = mobile_coords[quality_mask]

            if debug_enabled:
                logger.debug(
                    "%d CA pairs after quality filtering (%.1f <= B <= %.1f, occ >= %.1f).",
                    len(fixed_coords),
                    min_bfactor,
                    max_bfactor,
                    min_occ,
                )

            if len(fixed_coords) == 0:
                logger.warning(
                    "Mapping %s → %s: No aligned CA pairs left after quality filtering. Skipping this pair of chains.",
                    chain_id_fixed,
                    chain_id_mobile,
                )
                continue

        # Pool coordinates and chain IDs
        fixed_coords_list.append(fixed_coords)
        mobile_coords_list.append(mobile_coords)

    pooled_fixed_coords = np.vstack(fixed_coords_list) if fixed_coords_list else np.empty((0, 3))
    pooled_mobile_coords = np.vstack(mobile_coords_list) if mobile_coords_list else np.empty((0, 3))

    num_aligned = pooled_fixed_coords.shape[0]
    if num_aligned < 3:
        raise ValueError(f"Need at least 3 aligned CA pairs after pooling, found {pooled_fixed_coords.shape[0]}")

    if debug_enabled:
        logger.debug("\n-- Total number of aligned CA pairs across %d chains: %d --", len(chain_mapping), num_aligned)

    # Compute transformation
    if refine:
        rotation, translation, mask, rmsd = iterative_superpose(
            pooled_fixed_coords, pooled_mobile_coords, max_cycles=max_cycles, cutoff_factor=cutoff_factor
        )
        num_aligned = int(np.sum(mask))
    else:
        rotation, translation = superpose(pooled_fixed_coords, pooled_mobile_coords)
        mobile_transformed = pooled_mobile_coords @ rotation.T + translation
        rmsd = calculate_rmsd(pooled_fixed_coords, mobile_transformed)

    return rotation, translation, rmsd, num_aligned


def align_quaternary(
    fixed_chains: list[ProteinChain],
    mobile_chains: list[ProteinChain],
    fixed_seed: str | None = None,
    mobile_seed: str | None = None,
    distance_threshold: float = 8.0,
    refine: bool = False,
    cutoff_factor: float = 2.0,
    max_cycles: int = 5,
    filter: bool = False,
    min_bfactor: float = -np.inf,
    max_bfactor: float = np.inf,
    min_occ: float = -np.inf,
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, str]]:
    """Quaternary structure alignment with smart chain matching."""
    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    # Seed alignment
    fixed_chain_map = {chn.chain_id: chn for chn in fixed_chains}
    mobile_chain_map = {chn.chain_id: chn for chn in mobile_chains}
    if fixed_seed and fixed_seed not in fixed_chain_map:
        available = ", ".join(sorted(fixed_chain_map)) or "none"
        raise ValueError(f"Fixed seed chain '{fixed_seed}' not found. Available chains: {available}")
    if mobile_seed and mobile_seed not in mobile_chain_map:
        available = ", ".join(sorted(mobile_chain_map)) or "none"
        raise ValueError(f"Mobile seed chain '{mobile_seed}' not found. Available chains: {available}")
    fixed_chain = fixed_chain_map[fixed_seed] if fixed_seed else fixed_chains[0]
    mobile_chain = mobile_chain_map[mobile_seed] if mobile_seed else mobile_chains[0]

    if debug_enabled:
        logger.debug("-- Seed alignment: %s → %s --", fixed_chain.chain_id, mobile_chain.chain_id)

    rotation, translation, rmsd, num_aligned = align_two_chains(
        fixed_chain,
        mobile_chain,
        refine,
        cutoff_factor,
        max_cycles,
        filter,
        min_bfactor,
        max_bfactor,
        min_occ,
    )

    if debug_enabled:
        logger.debug("Aligned %d CA pairs with RMSD %.3f Å", num_aligned, rmsd)

    # Calculate all chain centers and pairwise distances.
    fixed_centers, fixed_valid_indices, fixed_invalid_ids = _get_valid_chain_centers(fixed_chains)
    mobile_centers, mobile_valid_indices, mobile_invalid_ids = _get_valid_chain_centers(mobile_chains)
    mobile_centers = mobile_centers @ rotation.T + translation
    distances = np.sqrt(np.sum((fixed_centers[:, np.newaxis, :] - mobile_centers[np.newaxis, :, :]) ** 2, axis=-1))

    if fixed_invalid_ids:
        logger.warning(
            "Skipping fixed chains with no finite CA coordinates: %s",
            ", ".join(fixed_invalid_ids),
        )
    if mobile_invalid_ids:
        logger.warning(
            "Skipping mobile chains with no finite CA coordinates: %s",
            ", ".join(mobile_invalid_ids),
        )

    if debug_enabled:
        logger.debug("\n-- Chain center distances after seed alignment. --")
        for i, row in enumerate(distances):
            for j, distance in enumerate(row):
                status = "✓" if distance <= distance_threshold else "✗"
                logger.debug(
                    "  %s ↔ %s: %.2f Å %s",
                    fixed_chains[fixed_valid_indices[i]].chain_id,
                    mobile_chains[mobile_valid_indices[j]].chain_id,
                    distance,
                    status,
                )
        logger.debug("")

    matched_idx = _match_chain_centers(distances, distance_threshold)
    chain_mapping = {
        fixed_chains[fixed_valid_indices[i]].chain_id: mobile_chains[mobile_valid_indices[j]].chain_id
        for i, j in matched_idx
    }

    logger.info(
        "Chain mapping after seed alignment: %s",
        ", ".join([f"{chn_id_1} → {chn_id_2}" for chn_id_1, chn_id_2 in chain_mapping.items()]),
    )

    rotation, translation, rmsd, num_aligned = align_mapped_chains(
        fixed_chain_map,
        mobile_chain_map,
        chain_mapping,
        refine,
        cutoff_factor,
        max_cycles,
        filter,
        min_bfactor,
        max_bfactor,
        min_occ,
    )

    return rotation, translation, rmsd, num_aligned, chain_mapping


def align_hungarian(
    fixed_chains: list[ProteinChain],
    mobile_chains: list[ProteinChain],
    fixed_seed: str | None = None,
    mobile_seed: str | None = None,
    distance_threshold: float = 8.0,
    refine: bool = False,
    cutoff_factor: float = 2.0,
    max_cycles: int = 5,
    filter: bool = False,
    min_bfactor: float = -np.inf,
    max_bfactor: float = np.inf,
    min_occ: float = -np.inf,
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, str]]:
    """Quaternary structure alignment using Hungarian chain matching.

    This follows the same flow as :func:`align_quaternary`:
    1. Align one seed chain pair to obtain an initial rigid-body transform.
    2. Transform mobile chain centers into the fixed reference frame.
    3. Build one-to-one chain mapping using global minimum-cost assignment.
    4. Recompute final transform from pooled coordinates across mapped chains.
    """
    if len(fixed_chains) == 0:
        raise ValueError("No fixed chains provided.")
    if len(mobile_chains) == 0:
        raise ValueError("No mobile chains provided.")

    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    fixed_chain_map = {chn.chain_id: chn for chn in fixed_chains}
    mobile_chain_map = {chn.chain_id: chn for chn in mobile_chains}
    if fixed_seed and fixed_seed not in fixed_chain_map:
        available = ", ".join(sorted(fixed_chain_map)) or "none"
        raise ValueError(f"Fixed seed chain '{fixed_seed}' not found. Available chains: {available}")
    if mobile_seed and mobile_seed not in mobile_chain_map:
        available = ", ".join(sorted(mobile_chain_map)) or "none"
        raise ValueError(f"Mobile seed chain '{mobile_seed}' not found. Available chains: {available}")
    fixed_chain = fixed_chain_map[fixed_seed] if fixed_seed else fixed_chains[0]
    mobile_chain = mobile_chain_map[mobile_seed] if mobile_seed else mobile_chains[0]

    if debug_enabled:
        logger.debug("-- Seed alignment: %s -> %s --", fixed_chain.chain_id, mobile_chain.chain_id)

    rotation, translation, rmsd, num_aligned = align_two_chains(
        fixed_chain,
        mobile_chain,
        refine,
        cutoff_factor,
        max_cycles,
        filter,
        min_bfactor,
        max_bfactor,
        min_occ,
    )

    if debug_enabled:
        logger.debug("Aligned %d CA pairs with RMSD %.3f A", num_aligned, rmsd)

    fixed_centers, fixed_valid_indices, fixed_invalid_ids = _get_valid_chain_centers(fixed_chains)
    mobile_centers, mobile_valid_indices, mobile_invalid_ids = _get_valid_chain_centers(mobile_chains)
    mobile_centers = mobile_centers @ rotation.T + translation
    distances = np.sqrt(np.sum((fixed_centers[:, np.newaxis, :] - mobile_centers[np.newaxis, :, :]) ** 2, axis=-1))

    if fixed_invalid_ids:
        logger.warning(
            "Skipping fixed chains with no finite CA coordinates: %s",
            ", ".join(fixed_invalid_ids),
        )
    if mobile_invalid_ids:
        logger.warning(
            "Skipping mobile chains with no finite CA coordinates: %s",
            ", ".join(mobile_invalid_ids),
        )

    if debug_enabled:
        logger.debug("\n-- Chain center distances after seed alignment. --")
        for i, row in enumerate(distances):
            for j, distance in enumerate(row):
                status = "ok" if distance <= distance_threshold else "skip"
                logger.debug(
                    "  %s <-> %s: %.2f A (%s)",
                    fixed_chains[fixed_valid_indices[i]].chain_id,
                    mobile_chains[mobile_valid_indices[j]].chain_id,
                    distance,
                    status,
                )
        logger.debug("")

    matched_idx = _match_chain_centers_hungarian(distances, distance_threshold)
    chain_mapping = {
        fixed_chains[fixed_valid_indices[i]].chain_id: mobile_chains[mobile_valid_indices[j]].chain_id
        for i, j in matched_idx
    }

    logger.info(
        "Chain mapping after seed alignment (Hungarian): %s",
        ", ".join([f"{chn_id_1} -> {chn_id_2}" for chn_id_1, chn_id_2 in chain_mapping.items()]),
    )

    rotation, translation, rmsd, num_aligned = align_mapped_chains(
        fixed_chain_map,
        mobile_chain_map,
        chain_mapping,
        refine,
        cutoff_factor,
        max_cycles,
        filter,
        min_bfactor,
        max_bfactor,
        min_occ,
    )

    return rotation, translation, rmsd, num_aligned, chain_mapping


def _match_chain_centers(distances: NDArray[np.floating], distance_threshold: float) -> list[tuple[int, int]]:
    """Greedy one-to-one chain matching based on center distances within a threshold."""
    masked = distances.copy()
    masked[~np.isfinite(masked)] = np.inf
    masked[masked > distance_threshold] = np.inf

    if not np.isfinite(masked).any():
        raise ValueError(
            f"No matching chains found. Adjust distance threshold (current value: {distance_threshold:.2f} Å)."
        )

    order = np.argsort(masked, axis=None)
    used_fixed = np.zeros(masked.shape[0], dtype=bool)
    used_mobile = np.zeros(masked.shape[1], dtype=bool)

    pairs: list[tuple[int, int]] = []
    for idx in order:
        i, j = np.unravel_index(idx, masked.shape)
        if not np.isfinite(masked[i, j]):
            break
        if used_fixed[i] or used_mobile[j]:
            continue
        pairs.append((i, j))
        used_fixed[i] = True
        used_mobile[j] = True

    return pairs


def _get_valid_chain_centers(
    chains: list[ProteinChain],
) -> tuple[NDArray[np.floating], list[int], list[str]]:
    """Return centers for chains that have at least one finite coordinate triplet."""
    valid_centers: list[NDArray[np.floating]] = []
    valid_indices: list[int] = []
    invalid_chain_ids: list[str] = []

    for idx, chain in enumerate(chains):
        finite_mask = np.all(np.isfinite(chain.coords), axis=1)
        if not np.any(finite_mask):
            invalid_chain_ids.append(chain.chain_id)
            continue
        center = np.mean(chain.coords[finite_mask], axis=0)
        valid_centers.append(center)
        valid_indices.append(idx)

    if len(valid_centers) == 0:
        raise ValueError("No chains with finite CA coordinates available for quaternary matching.")

    return np.array(valid_centers, dtype=float), valid_indices, invalid_chain_ids


def _hungarian_minimize(cost: NDArray[np.floating]) -> list[tuple[int, int]]:
    """Compute minimum-cost assignment using the Hungarian algorithm.

    Args:
        cost: Cost matrix of shape (n_rows, n_cols). All values must be finite.

    Returns:
        List of (row_index, col_index) assignments. One assignment per row.
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []

    transposed = False
    work_cost = cost
    if n_rows > n_cols:
        work_cost = cost.T
        n_rows, n_cols = work_cost.shape
        transposed = True

    if not np.all(np.isfinite(work_cost)):
        raise ValueError("Hungarian algorithm requires finite costs.")

    u = np.zeros(n_rows + 1, dtype=float)
    v = np.zeros(n_cols + 1, dtype=float)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=float)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = work_cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: list[tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            row_idx = p[j] - 1
            col_idx = j - 1
            if transposed:
                assignment.append((col_idx, row_idx))
            else:
                assignment.append((row_idx, col_idx))

    assignment.sort(key=lambda x: x[0])
    return assignment


def _match_chain_centers_hungarian(distances: NDArray[np.floating], distance_threshold: float) -> list[tuple[int, int]]:
    """Threshold-aware global chain matching using the Hungarian algorithm."""
    valid_mask = np.isfinite(distances) & (distances <= distance_threshold)
    if not valid_mask.any():
        raise ValueError(
            f"No matching chains found. Adjust distance threshold (current value: {distance_threshold:.2f} A)."
        )

    n_fixed, n_mobile = distances.shape
    n_total = n_fixed + n_mobile

    valid_distances = distances[valid_mask]
    max_valid = float(np.max(valid_distances))
    penalty = max(distance_threshold + 1.0, max_valid + 1.0)
    invalid_cost = penalty * 100.0

    cost = np.zeros((n_total, n_total), dtype=float)
    real_block = np.where(valid_mask, distances, invalid_cost)
    cost[:n_fixed, :n_mobile] = real_block

    # Costs for matching real chains to dummy slots (unmatched).
    cost[:n_fixed, n_mobile:] = penalty
    cost[n_fixed:, :n_mobile] = penalty
    cost[n_fixed:, n_mobile:] = 0.0

    assignment = _hungarian_minimize(cost)

    pairs: list[tuple[int, int]] = []
    for fixed_idx, col_idx in assignment:
        if fixed_idx >= n_fixed:
            continue
        if col_idx >= n_mobile:
            continue
        if valid_mask[fixed_idx, col_idx]:
            pairs.append((fixed_idx, col_idx))

    if len(pairs) == 0:
        raise ValueError(
            f"No matching chains found. Adjust distance threshold (current value: {distance_threshold:.2f} A)."
        )

    return pairs
