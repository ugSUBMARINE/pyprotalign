"""Coordinate transformation operations."""

import gemmi
import numpy as np
from numpy.typing import NDArray


def apply_transformation(
    structure: gemmi.Structure,
    rotation: NDArray[np.floating],
    translation: NDArray[np.floating],
) -> None:
    """Apply rigid-body transformation to all atoms in structure.

    Transforms all atoms in-place using: new_pos = old_pos @ rotation.T + translation

    Args:
        structure: Input gemmi Structure (modified in-place)
        rotation: Rotation matrix, shape (3, 3)
        translation: Translation vector, shape (3,)
    """
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Get current position
                    pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])

                    # Apply transformation
                    new_pos = pos @ rotation.T + translation

                    # Update atom position
                    atom.pos.x = float(new_pos[0])
                    atom.pos.y = float(new_pos[1])
                    atom.pos.z = float(new_pos[2])


def generate_conflict_free_chain_map(
    structure: gemmi.Structure,
    chain_pairs: list[tuple[str, str]],
) -> dict[str, str]:
    """Generate chain rename map that avoids conflicts with unaligned chains.

    For quaternary alignment with chain renaming, this ensures that renaming
    aligned chains doesn't create duplicate chain names with unaligned chains.
    Uses swap strategy: if renaming mobile chain A to B conflicts with existing
    unaligned chain B, adds B->A swap to the map.

    Args:
        structure: Mobile structure
        chain_pairs: List of (fixed_name, mobile_name) tuples from alignment

    Returns:
        Complete rename mapping with conflict resolution via swaps
    """
    # Get all chain names in mobile structure
    all_mobile_chains = {chain.name for model in structure for chain in model}

    # Get aligned mobile chain names (sources being renamed)
    aligned_mobile_chains = {mobile_name for _, mobile_name in chain_pairs}

    # Build rename map with conflict resolution
    chain_map: dict[str, str] = {}

    for fixed_name, mobile_name in chain_pairs:
        # Skip identity renames (chain keeps same name)
        if fixed_name == mobile_name:
            continue

        # Add primary rename: mobile -> fixed
        chain_map[mobile_name] = fixed_name

        # Check for conflict: does fixed_name already exist as an unaligned chain?
        # An unaligned chain is one that exists but is NOT being renamed (not in aligned_mobile_chains)
        if fixed_name in all_mobile_chains and fixed_name not in aligned_mobile_chains:
            # Add swap: fixed_name -> mobile_name to move it out of the way
            chain_map[fixed_name] = mobile_name

    return chain_map


def rename_chains(structure: gemmi.Structure, chain_map: dict[str, str]) -> None:
    """Rename chains in structure based on mapping.

    Uses temporary names to avoid collisions during renaming.
    Skips identity renames where old name equals new name.

    Args:
        structure: Input gemmi Structure (modified in-place)
        chain_map: Mapping from old chain names to new chain names
    """
    if not chain_map:
        return

    # Filter out identity renames
    filtered_map = {old: new for old, new in chain_map.items() if old != new}
    if not filtered_map:
        return

    # Step 1: Rename to temporary names to avoid collisions
    temp_map: dict[str, str] = {}
    for model in structure:
        for i, chain in enumerate(model):
            if chain.name in filtered_map:
                temp_name = f"__TEMP_{i}__"
                temp_map[temp_name] = filtered_map[chain.name]
                chain.name = temp_name

    # Step 2: Rename from temporary to final names
    for model in structure:
        for chain in model:
            if chain.name in temp_map:
                chain.name = temp_map[chain.name]
