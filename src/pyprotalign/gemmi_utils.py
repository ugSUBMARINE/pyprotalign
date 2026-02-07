"""Utility functions for working with Gemmi library in pyprotalign package."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from textwrap import wrap

import gemmi
import numpy as np
from numpy.typing import NDArray

from .chain import ProteinChain

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def align_sequences(seq_1: str, seq_2: str) -> tuple[tuple[int | None, int | None], ...]:
    """Align two sequences and return paired indices.

    Args:
        seq_1: First sequence (fixed)
        seq_2: Second sequence (mobile)

    Returns:
        Tuple of paired indices. Each tuple contains (seq1_idx, seq2_idx).
        None indicates gap at that position.

    Example:
        For alignment:
            seq_1: AC-DEFG
            seq_2: ACYEDF-
        Returns: ((0,0), (1,1), (None,2), (2,3), (3,4), (4,5), (5,None))

    Note:
        Results are cached using LRU cache (maxsize=128) for performance
        in multi-chain and batch alignment scenarios.
    """
    result = gemmi.align_string_sequences(
        gemmi.expand_one_letter_sequence(seq_1, gemmi.ResidueKind.AA),
        gemmi.expand_one_letter_sequence(seq_2, gemmi.ResidueKind.AA),
        [],
        gemmi.AlignmentScoring("b"),
    )

    # Get aligned sequences from formatted output
    # Format is: seq1_aligned\nmatch_string\nseq2_aligned\n
    formatted = result.formatted(seq_1, seq_2)
    aligned_1, matches, aligned_2, _ = formatted.split("\n")

    # log alignment for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("\nSequence alignment:\n")
        for line_1, match_line, line_2 in zip(
            wrap(aligned_1, width=60), wrap(matches, width=60), wrap(aligned_2, width=60), strict=True
        ):
            logger.debug(" Fixed: %s", line_1)
            logger.debug("        %s", match_line)
            logger.debug("Mobile: %s\n", line_2)
        logger.debug("Score: %d", result.score)
        logger.debug("Indentity: %.1f%%\n", result.calculate_identity())

    pairs: list[tuple[int | None, int | None]] = []
    idx_1 = 0
    idx_2 = 0

    # Parse aligned sequences to build correspondence
    for aa_1, aa_2 in zip(aligned_1, aligned_2, strict=True):
        if aa_1 == "-":
            # Gap in seq1
            pairs.append((None, idx_2))
            idx_2 += 1
        elif aa_2 == "-":
            # Gap in seq2
            pairs.append((idx_1, None))
            idx_1 += 1
        else:
            # Match or mismatch
            pairs.append((idx_1, idx_2))
            idx_1 += 1
            idx_2 += 1

    return tuple(pairs)


def align_structures(
    fixed_st: gemmi.Structure,
    mobile_st: gemmi.Structure,
    fixed_chain_id: str | None = None,
    mobile_chain_id: str | None = None,
    cycles: int = 0,
    cutoff: float = 2.0,
) -> tuple[gemmi.Structure, float, int]:
    """Perform simple one-chain-to-one-chain alignment using gemmi's built-in function.

    Args:
        fixed_st: Fixed structure
        mobile_st: Mobile structure - will be modified in place
        args: Parsed command-line arguments

    Returns:
        Tuple of (transformed mobile structure, RMSD, info string for reporting)

    Note:
        Internal sequence alignment yields much smaller numbers of CA pairs ('sup.count')
        than the alignment method used in the custom functions. Reasons are unclear.
    """
    # get chains, default to first chain if not specified
    fixed_chain = fixed_st[0][0] if not fixed_chain_id else fixed_st[0][fixed_chain_id]
    mobile_chain = mobile_st[0][0] if not mobile_chain_id else mobile_st[0][mobile_chain_id]

    fixed_pol = fixed_chain.get_polymer()
    mobile_pol = mobile_chain.get_polymer()
    ptype = fixed_pol.check_polymer_type()

    logger.info("Fixed:  chain %s, %d residues", fixed_chain.name, len(fixed_pol))
    logger.info("Mobile: chain %s, %d residues", mobile_chain.name, len(mobile_pol))

    sup = gemmi.calculate_superposition(
        fixed_pol,
        mobile_pol,
        ptype,
        sel=gemmi.SupSelect.CaP,
        trim_cycles=cycles,
        trim_cutoff=cutoff,
    )

    logger.info("Aligned: %d CA atom pairs", sup.count)
    logger.info("RMSD: %.3f Å", sup.rmsd)

    for model in mobile_st:
        model.transform_pos_and_adp(sup.transform)

    return mobile_st, sup.rmsd, sup.count


def create_chain(chain: gemmi.Chain) -> ProteinChain:
    """Create ProteinChain from gemmi.Chain object.

    Args:
        chain: gemmi Chain object (must be a protein chain)

    Returns:
        ProteinChain instance

    Raises:
        ValueError: If chain is not a protein chain or has no residues

    Note:
        Alternate locations (altlocs) are handled by Gemmi's default atom
        selection behavior when iterating residues. This may differ from
        Biopython's behavior (which often selects the higher-occupancy altloc),
        leading to small discrepancies in coordinates, B-factors, and occupancies.
    """
    # Get polymer residues
    polymer = chain.get_polymer()
    if len(polymer) == 0 or polymer.check_polymer_type() != gemmi.PolymerType.PeptideL:
        raise ValueError(f"Chain '{chain.name}' is not a protein chain")

    # Extract one-letter sequence, removing gaps that are placed at chain breaks
    sequence = polymer.make_one_letter_sequence().replace("-", "")
    n_residues = len(sequence)

    # Initialize arrays with NaN (for missing atoms)
    coords = np.full((n_residues, 3), np.nan, dtype=float)
    b_factors = np.full(n_residues, np.nan, dtype=float)
    occupancies = np.full(n_residues, np.nan, dtype=float)

    # Extract CA atom data
    for i, residue in enumerate(polymer):
        ca = residue.get_ca()
        if ca:
            coords[i] = [ca.pos.x, ca.pos.y, ca.pos.z]
            b_factors[i] = ca.b_iso
            occupancies[i] = ca.occ

    return ProteinChain(
        chain_id=chain.name,
        sequence=sequence,
        coords=coords,
        b_factors=b_factors,
        occupancies=occupancies,
    )


def get_first_protein_chain(model: gemmi.Model) -> ProteinChain:
    """Extract first protein chain from structure.

    Args:
        model: Input gemmi Model

    Returns:
        First protein chain found in model

    Raises:
        ValueError: If no protein chain found
    """
    for chain in model:
        try:
            return create_chain(chain)
        except ValueError:
            continue  # Skip non-protein chains

    raise ValueError("No protein chain found in structure")


def get_chain(model: gemmi.Model, chain_id: str | None = None) -> ProteinChain:
    """Get chain by ID or first protein chain.

    Args:
        model: Input gemmi Model
        chain_id: Chain identifier (e.g. "A", "B"). If None, returns first protein chain.

    Returns:
        Requested chain

    Raises:
        ValueError: If chain_id specified but not found, or chain is not protein
    """
    if chain_id is None:
        return get_first_protein_chain(model)

    for chain in model:
        try:
            protein_chain = create_chain(chain)
            if protein_chain.chain_id == chain_id:
                return protein_chain
        except ValueError:
            continue  # Skip non-protein chains

    raise ValueError(f"No protein chain '{chain_id}' found in structure")


def get_all_protein_chains(model: gemmi.Model) -> list[ProteinChain]:
    """Get all protein chains from model.

    Args:
        model: Input gemmi Model

    Returns:
        List of all protein chains in model

    Raises:
        ValueError: If no protein chains found in model
    """
    chains = []
    for chain in model:
        try:
            protein_chain = create_chain(chain)
            chains.append(protein_chain)
        except ValueError:
            continue  # Skip non-protein chains

    if len(chains) == 0:
        raise ValueError("No protein chains found in model")

    return chains


def load_structure(path: str | Path) -> gemmi.Structure:
    """Load a protein structure from PDB or mmCIF file.

    Args:
        path: Path to structure file (PDB or mmCIF format)

    Returns:
        Loaded gemmi Structure object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If structure has no models
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    structure = gemmi.read_structure(str(file_path))

    if len(structure) == 0:
        raise ValueError(f"Structure has no models: {path}")

    # Setup entities to identify polymer chains
    structure.setup_entities()

    return structure


def write_structure(structure: gemmi.Structure, path: str | Path) -> None:
    """Write structure to file with format auto-detection.

    Args:
        structure: gemmi Structure to write
        path: Output file path (.pdb or .cif extension)
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".cif":
        # Write mmCIF
        doc = structure.make_mmcif_document()
        doc.write_file(str(file_path))
    else:
        # Write PDB (default)
        structure.write_pdb(str(file_path))


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
    chain_mapping: dict[str, str],
) -> dict[str, str]:
    """Generate chain rename map that avoids conflicts with unaligned chains.

    For quaternary alignment with chain renaming, this ensures that renaming
    aligned chains doesn't create duplicate chain names with unaligned chains.
    Uses swap strategy: if renaming mobile chain A to B conflicts with existing
    unaligned chain B, adds B->A swap to the map.

    Args:
        structure: Mobile structure
        chain_mapping: Dictionary with chain mappings (fixed to mobile) from alignment

    Returns:
        Complete rename mapping with conflict resolution via swaps
    """
    # Get all chain names in mobile structure
    all_mobile_chains = {chain.name for chain in structure[0]}

    # Get aligned mobile chain names (sources being renamed)
    aligned_mobile_chains = {mobile_name for mobile_name in chain_mapping.values()}

    # Build rename map with conflict resolution
    chain_rename_map: dict[str, str] = {}

    for fixed_name, mobile_name in chain_mapping.items():
        # Skip identity renames (chain keeps same name)
        if fixed_name == mobile_name:
            continue

        # Add primary rename: mobile -> fixed
        chain_rename_map[mobile_name] = fixed_name

        # Check for conflict: does fixed_name already exist as an unaligned chain?
        # An unaligned chain is one that exists but is NOT being renamed (not in aligned_mobile_chains)
        if fixed_name in all_mobile_chains and fixed_name not in aligned_mobile_chains:
            # Add swap: fixed_name -> mobile_name to move it out of the way
            chain_rename_map[fixed_name] = mobile_name

    return chain_rename_map


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
    for model_index, model in enumerate(structure):
        for chain_index, chain in enumerate(model):
            if chain.name in filtered_map:
                temp_name = f"__CHN_{model_index}_{chain_index}__"
                temp_map[temp_name] = filtered_map[chain.name]
                chain.name = temp_name

    # Step 2: Rename from temporary to final names
    for model in structure:
        for chain in model:
            if chain.name in temp_map:
                chain.name = temp_map[chain.name]
