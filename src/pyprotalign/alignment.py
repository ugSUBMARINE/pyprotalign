"""Sequence alignment operations."""

import logging

import numpy as np

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
    chn_1: ProteinChain, chn_2: ProteinChain, pairs: list[tuple[int | None, int | None]]
) -> tuple[list[int], list[int]]:
    """Helper to extract aligned residue indices from sequence alignment pairs.

    Args:
        chn_1: First ProteinChain
        chn_2: Second ProteinChain
        pairs: List of (idx1, idx2) from align_sequences

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

    # Calculate all chain centers and pairwise distances
    fixed_centers = np.array([chn.coords.mean(axis=0) for chn in fixed_chains], dtype=float)
    mobile_centers = (
        np.array([chn.coords.mean(axis=0) for chn in mobile_chains], dtype=float) @ rotation.T + translation
    )
    distances = np.sqrt(np.sum((fixed_centers[:, np.newaxis, :] - mobile_centers[np.newaxis, :, :]) ** 2, axis=-1))

    if debug_enabled:
        logger.debug("\n-- Chain center distances after seed alignemnt. --")
        for i, row in enumerate(distances):
            for j, distance in enumerate(row):
                status = "✓" if distance <= distance_threshold else "✗"
                logger.debug(
                    "  %s ↔ %s: %.2f Å %s", fixed_chains[i].chain_id, mobile_chains[j].chain_id, distance, status
                )
        logger.debug("")

    matched_idx = np.argwhere(distances <= distance_threshold)
    n_rows = matched_idx.shape[0]
    if n_rows == 0:
        raise ValueError(
            f"No matching chains found. Adjust distance threshold (current value: {distance_threshold:.2f} Å)."
        )
    if n_rows > len(fixed_chains) or n_rows > len(mobile_chains):
        raise ValueError(
            f"Inconsistent chain mapping. Adjust distance threshold (current value: {distance_threshold:.2f} Å)."
        )

    chain_mapping = {fixed_chains[i].chain_id: mobile_chains[j].chain_id for i, j in matched_idx}

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
