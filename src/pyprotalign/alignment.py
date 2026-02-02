"""Sequence alignment operations."""

import copy

import gemmi
import numpy as np

from .kabsch import superpose
from .refine import iterative_superpose
from .selection import (
    compute_chain_center,
    extract_ca_atoms,
    extract_sequence,
    get_all_protein_chains,
    get_chain,
)
from .transform import apply_transformation


def align_sequences(seq1: str, seq2: str) -> list[tuple[int | None, int | None]]:
    """Align two sequences and return paired indices.

    Args:
        seq1: First sequence (fixed)
        seq2: Second sequence (mobile)

    Returns:
        List of paired indices. Each tuple contains (seq1_idx, seq2_idx).
        None indicates gap at that position.

    Example:
        For alignment:
            seq1: AC-DEFG
            seq2: ACDXFG-
        Returns: [(0,0), (1,1), (None,2), (2,3), (3,4), (4,5), (5,None)]
    """
    result = gemmi.align_string_sequences(list(seq1), list(seq2), [0] * len(seq2))

    # Get aligned sequences from formatted output
    # Format is: seq1_aligned\nmatch_string\nseq2_aligned\n
    formatted = result.formatted(seq1, seq2)
    lines = formatted.strip().split("\n")
    aligned1 = lines[0]
    aligned2 = lines[2]

    pairs: list[tuple[int | None, int | None]] = []
    idx1 = 0
    idx2 = 0

    # Parse aligned sequences to build correspondence
    for i in range(len(aligned1)):
        if aligned1[i] == "-":
            # Gap in seq1
            pairs.append((None, idx2))
            idx2 += 1
        elif aligned2[i] == "-":
            # Gap in seq2
            pairs.append((idx1, None))
            idx1 += 1
        else:
            # Match or mismatch
            pairs.append((idx1, idx2))
            idx1 += 1
            idx2 += 1

    return pairs


def align_multi_chain(
    fixed_structure: gemmi.Structure, mobile_structure: gemmi.Structure
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align multiple chains by matching chain IDs and pooling coordinates.

    Matches chains by name (A-A, B-B, etc.) and aligns sequences for each pair.
    Pools all aligned CA atom coordinates across chains.

    Args:
        fixed_structure: Fixed structure with setup entities
        mobile_structure: Mobile structure with setup entities

    Returns:
        Tuple of (fixed_coords, mobile_coords, chain_ids):
        - fixed_coords: Nx3 array of CA coordinates from fixed structure
        - mobile_coords: Nx3 array of CA coordinates from mobile structure
        - chain_ids: List of chain IDs for each coordinate pair

    Raises:
        ValueError: If no matching protein chains found or fewer than 3 aligned pairs
    """
    fixed_chains = get_all_protein_chains(fixed_structure)
    mobile_chains = get_all_protein_chains(mobile_structure)

    # Build chain name -> chain mapping
    fixed_chain_map = {chain.name: chain for chain in fixed_chains}
    mobile_chain_map = {chain.name: chain for chain in mobile_chains}

    # Find matching chain names
    common_chains = set(fixed_chain_map.keys()) & set(mobile_chain_map.keys())

    if not common_chains:
        raise ValueError("No matching protein chains found between structures")

    # Collect coordinates across all matching chains
    fixed_coords_list = []
    mobile_coords_list = []
    chain_ids = []

    for chain_name in sorted(common_chains):  # Sort for deterministic order
        fixed_chain = fixed_chain_map[chain_name]
        mobile_chain = mobile_chain_map[chain_name]

        # Extract sequences
        fixed_seq = extract_sequence(fixed_chain)
        mobile_seq = extract_sequence(mobile_chain)

        # Align sequences
        pairs = align_sequences(fixed_seq, mobile_seq)

        # Extract CA atoms
        fixed_cas = extract_ca_atoms(fixed_chain)
        mobile_cas = extract_ca_atoms(mobile_chain)

        # Build coordinate arrays for this chain
        for fix_idx, mob_idx in pairs:
            if fix_idx is not None and mob_idx is not None:
                # Check that CA atoms exist
                if fix_idx < len(fixed_cas) and mob_idx < len(mobile_cas):
                    fixed_ca = fixed_cas[fix_idx]
                    mobile_ca = mobile_cas[mob_idx]
                    fixed_coords_list.append([fixed_ca.pos.x, fixed_ca.pos.y, fixed_ca.pos.z])
                    mobile_coords_list.append([mobile_ca.pos.x, mobile_ca.pos.y, mobile_ca.pos.z])
                    chain_ids.append(chain_name)

    if len(fixed_coords_list) < 3:
        raise ValueError(f"Need at least 3 aligned CA pairs, found {len(fixed_coords_list)}")

    fixed_coords = np.array(fixed_coords_list)
    mobile_coords = np.array(mobile_coords_list)

    return fixed_coords, mobile_coords, chain_ids


def align_quaternary(
    fixed_structure: gemmi.Structure,
    mobile_structure: gemmi.Structure,
    distance_threshold: float,
    seed_fixed_chain: str | None = None,
    seed_mobile_chain: str | None = None,
    refine: bool = False,
    cutoff_factor: float = 2.0,
    max_cycles: int = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Quaternary structure alignment with smart chain matching.

    Aligns structures by first aligning seed chains, then matching remaining
    chains by proximity after initial transformation. Returns pooled coordinates
    and chain pairing information.

    Args:
        fixed_structure: Fixed structure with setup entities
        mobile_structure: Mobile structure with setup entities
        distance_threshold: Max distance (Å) between chain centers for matching
        seed_fixed_chain: Chain ID for seed alignment in fixed (None = first protein chain)
        seed_mobile_chain: Chain ID for seed alignment in mobile (None = first protein chain)
        refine: Enable iterative refinement for both seed and final alignment
        cutoff_factor: Outlier rejection cutoff for refinement
        max_cycles: Maximum refinement cycles
        verbose: Print detailed chain matching information

    Returns:
        Tuple of (fixed_coords, mobile_coords, chain_pairs):
        - fixed_coords: Nx3 array of CA coordinates from fixed structure
        - mobile_coords: Nx3 array of CA coordinates from mobile structure
        - chain_pairs: List of (fixed_name, mobile_name) tuples for matched chains

    Raises:
        ValueError: If fewer than 3 aligned pairs or no matching chains found
    """
    # Get all protein chains
    fixed_chains = get_all_protein_chains(fixed_structure)
    mobile_chains = get_all_protein_chains(mobile_structure)

    if len(fixed_chains) == 0 or len(mobile_chains) == 0:
        raise ValueError("Both structures must have at least one protein chain")

    # Select seed chains
    seed_fixed = get_chain(fixed_structure, seed_fixed_chain)
    seed_mobile = get_chain(mobile_structure, seed_mobile_chain)

    if verbose:
        print(f"Seed alignment: {seed_fixed.name} → {seed_mobile.name}")

    # Align seed chains to establish initial transformation
    fixed_seq = extract_sequence(seed_fixed)
    mobile_seq = extract_sequence(seed_mobile)
    pairs = align_sequences(fixed_seq, mobile_seq)

    fixed_cas = extract_ca_atoms(seed_fixed)
    mobile_cas = extract_ca_atoms(seed_mobile)

    # Extract seed coordinates
    seed_fixed_coords = []
    seed_mobile_coords = []
    for fix_idx, mob_idx in pairs:
        if fix_idx is not None and mob_idx is not None:
            if fix_idx < len(fixed_cas) and mob_idx < len(mobile_cas):
                fixed_ca = fixed_cas[fix_idx]
                mobile_ca = mobile_cas[mob_idx]
                seed_fixed_coords.append([fixed_ca.pos.x, fixed_ca.pos.y, fixed_ca.pos.z])
                seed_mobile_coords.append([mobile_ca.pos.x, mobile_ca.pos.y, mobile_ca.pos.z])

    if len(seed_fixed_coords) < 3:
        raise ValueError(f"Need at least 3 aligned CA pairs in seed chains, found {len(seed_fixed_coords)}")

    seed_fixed_arr = np.array(seed_fixed_coords)
    seed_mobile_arr = np.array(seed_mobile_coords)

    # Compute initial transformation
    if refine:
        initial_rotation, initial_translation, _, _ = iterative_superpose(
            seed_fixed_arr, seed_mobile_arr, max_cycles=max_cycles, cutoff_factor=cutoff_factor, verbose=verbose
        )
    else:
        initial_rotation, initial_translation = superpose(seed_fixed_arr, seed_mobile_arr)

    # Create temporary copy of mobile structure and apply initial transformation
    mobile_temp = copy.deepcopy(mobile_structure)
    apply_transformation(mobile_temp, initial_rotation, initial_translation)

    if verbose:
        print("Chain center distances after seed alignment:")

    # Match chains by proximity
    chain_pairs: list[tuple[str, str]] = [(seed_fixed.name, seed_mobile.name)]
    used_mobile_names = {seed_mobile.name}

    # Build chain maps for efficient lookup
    fixed_chain_map = {chain.name: chain for chain in fixed_chains}
    mobile_chain_map = {chain.name: chain for chain in mobile_chains}
    mobile_temp_map = {chain.name: chain for chain in get_all_protein_chains(mobile_temp)}

    # Match remaining chains
    for fixed_chain in fixed_chains:
        if fixed_chain.name == seed_fixed.name:
            continue  # Already paired

        # Compute fixed chain center
        fixed_center = compute_chain_center(fixed_chain)

        # Find closest unmatched mobile chain
        best_mobile_name = None
        best_distance = float("inf")

        for mobile_chain_name in mobile_chain_map.keys():
            if mobile_chain_name in used_mobile_names:
                continue

            mobile_temp_chain = mobile_temp_map[mobile_chain_name]
            mobile_center = compute_chain_center(mobile_temp_chain)
            distance = float(np.linalg.norm(fixed_center - mobile_center))

            if verbose:
                status = "✓" if distance <= distance_threshold else "✗"
                print(f"  {fixed_chain.name} ↔ {mobile_chain_name}: {distance:.2f} Å {status}")

            if distance < best_distance:
                best_distance = distance
                best_mobile_name = mobile_chain_name

        # Add pair if within threshold
        if best_mobile_name is not None and best_distance <= distance_threshold:
            chain_pairs.append((fixed_chain.name, best_mobile_name))
            used_mobile_names.add(best_mobile_name)

    if len(chain_pairs) == 0:
        raise ValueError("No chain pairs matched")

    # Pool coordinates from all matched chains (using original untransformed structures)
    fixed_coords_list = []
    mobile_coords_list = []

    for fixed_name, mobile_name in chain_pairs:
        fixed_chain = fixed_chain_map[fixed_name]
        mobile_chain = mobile_chain_map[mobile_name]

        # Extract sequences and align
        fixed_seq = extract_sequence(fixed_chain)
        mobile_seq = extract_sequence(mobile_chain)
        pairs = align_sequences(fixed_seq, mobile_seq)

        # Extract CA atoms
        fixed_cas = extract_ca_atoms(fixed_chain)
        mobile_cas = extract_ca_atoms(mobile_chain)

        # Pool coordinates
        for fix_idx, mob_idx in pairs:
            if fix_idx is not None and mob_idx is not None:
                if fix_idx < len(fixed_cas) and mob_idx < len(mobile_cas):
                    fixed_ca = fixed_cas[fix_idx]
                    mobile_ca = mobile_cas[mob_idx]
                    fixed_coords_list.append([fixed_ca.pos.x, fixed_ca.pos.y, fixed_ca.pos.z])
                    mobile_coords_list.append([mobile_ca.pos.x, mobile_ca.pos.y, mobile_ca.pos.z])

    if len(fixed_coords_list) < 3:
        raise ValueError(f"Need at least 3 aligned CA pairs, found {len(fixed_coords_list)}")

    fixed_coords = np.array(fixed_coords_list)
    mobile_coords = np.array(mobile_coords_list)

    return fixed_coords, mobile_coords, chain_pairs
