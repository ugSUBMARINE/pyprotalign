"""Chain and atom selection operations."""

import gemmi
import numpy as np
from numpy.typing import NDArray


def get_first_protein_chain(structure: gemmi.Structure) -> gemmi.Chain:
    """Extract first protein chain from structure.

    Args:
        structure: Input gemmi Structure

    Returns:
        First protein chain found in first model

    Raises:
        ValueError: If no protein chain found
    """
    model = structure[0]
    for chain in model:
        # Check polymer residues
        polymer = chain.get_polymer()
        if len(polymer) > 0:
            return chain

    raise ValueError("No protein chain found in structure")


def get_chain(structure: gemmi.Structure, chain_id: str | None = None) -> gemmi.Chain:
    """Get chain by ID or first protein chain.

    Args:
        structure: Input gemmi Structure
        chain_id: Chain identifier (e.g. "A", "B"). If None, returns first protein chain.

    Returns:
        Requested chain

    Raises:
        ValueError: If chain_id specified but not found, or chain is not protein
    """
    if chain_id is None:
        return get_first_protein_chain(structure)

    model = structure[0]
    chain = model.find_chain(chain_id)

    if chain is None:
        raise ValueError(f"Chain '{chain_id}' not found in structure")

    polymer = chain.get_polymer()
    if len(polymer) == 0:
        raise ValueError(f"Chain '{chain_id}' is not a protein chain")

    return chain


def get_all_protein_chains(structure: gemmi.Structure) -> list[gemmi.Chain]:
    """Get all protein chains from structure.

    Args:
        structure: Input gemmi Structure

    Returns:
        List of all protein chains in first model
    """
    chains = []
    model = structure[0]
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) > 0:
            chains.append(chain)
    return chains


def extract_sequence(chain: gemmi.Chain) -> str:
    """Extract single-letter amino acid sequence from chain.

    Args:
        chain: Input gemmi Chain

    Returns:
        Single-letter amino acid sequence
    """
    polymer = chain.get_polymer()
    residue_names = [res.name for res in polymer]
    return gemmi.one_letter_code(residue_names)


def extract_ca_atoms(chain: gemmi.Chain) -> list[gemmi.Atom]:
    """Extract CA atoms from chain.

    Args:
        chain: Input gemmi Chain

    Returns:
        List of CA atoms (one per residue, skips residues without CA)
    """
    ca_atoms = []
    polymer = chain.get_polymer()
    for residue in polymer:
        ca = residue.find_atom("CA", "*")
        if ca:
            ca_atoms.append(ca)
    return ca_atoms


def extract_ca_atoms_by_residue(chain: gemmi.Chain) -> list[gemmi.Atom | None]:
    """Extract CA atoms aligned to polymer residue indices.

    Args:
        chain: Input gemmi Chain

    Returns:
        List of CA atoms or None, one entry per polymer residue
    """
    polymer = chain.get_polymer()
    ca_atoms: list[gemmi.Atom | None] = []
    for residue in polymer:
        ca = residue.find_atom("CA", "*")
        ca_atoms.append(ca if ca else None)
    return ca_atoms


def compute_chain_center(chain: gemmi.Chain) -> NDArray[np.floating]:
    """Compute geometric center of chain from CA atoms.

    Args:
        chain: Input gemmi Chain

    Returns:
        Center coordinates as (3,) array

    Raises:
        ValueError: If chain has no CA atoms
    """
    ca_atoms = extract_ca_atoms(chain)

    if len(ca_atoms) == 0:
        raise ValueError(f"Chain '{chain.name}' has no CA atoms")

    coords = np.array([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in ca_atoms])
    center: NDArray[np.floating] = np.mean(coords, axis=0)
    return center


def filter_ca_atoms_by_quality(
    ca_atoms: list[gemmi.Atom | None],
    min_plddt: float | None = None,
    max_bfactor: float | None = None,
) -> NDArray[np.bool_]:
    """Filter CA atoms by B-factor or pLDDT quality metric.

    Args:
        ca_atoms: List of CA atoms or None for missing atoms
        min_plddt: Minimum pLDDT threshold (atoms with b_iso < threshold filtered out)
        max_bfactor: Maximum B-factor threshold (atoms with b_iso > threshold filtered out)

    Returns:
        Boolean mask array (True = keep, False = filter out)

    Raises:
        ValueError: If both min_plddt and max_bfactor are specified, or if neither is specified
    """
    # Validate mutual exclusivity
    if min_plddt is not None and max_bfactor is not None:
        raise ValueError("min_plddt and max_bfactor are mutually exclusive")
    if min_plddt is None and max_bfactor is None:
        raise ValueError("Either min_plddt or max_bfactor must be specified")

    # Initialize mask to False (filter out) and set to True for atoms that pass the filter
    mask = np.zeros(len(ca_atoms), dtype=bool)

    for i, atom in enumerate(ca_atoms):
        # Skip missing atoms (None)
        if atom is None:
            continue

        if min_plddt is not None:
            # pLDDT filter: keep if >= threshold
            mask[i] = atom.b_iso >= min_plddt
        elif max_bfactor is not None:
            # B-factor filter: keep if <= threshold
            mask[i] = atom.b_iso <= max_bfactor

    return mask


def filter_ca_pairs_by_quality(
    fixed_cas: list[gemmi.Atom | None],
    mobile_cas: list[gemmi.Atom | None],
    min_plddt: float | None = None,
    max_bfactor: float | None = None,
) -> NDArray[np.bool_]:
    """Filter CA atom pairs by B-factor or pLDDT quality metric.

    Returns combined mask where both atoms in a pair must pass the filter.
    Preserves index relationship between fixed and mobile atom lists.

    Args:
        fixed_cas: List of CA atoms from fixed structure
        mobile_cas: List of CA atoms from mobile structure
        min_plddt: Minimum pLDDT threshold (atoms with b_iso < threshold filtered out)
        max_bfactor: Maximum B-factor threshold (atoms with b_iso > threshold filtered out)

    Returns:
        Boolean mask array (True = both pass, False = at least one fails)

    Raises:
        ValueError: If lists have different lengths, or if both/neither filter specified
    """
    if len(fixed_cas) != len(mobile_cas):
        raise ValueError(f"CA atom lists must have same length: {len(fixed_cas)} vs {len(mobile_cas)}")

    # Get individual masks
    fixed_mask = filter_ca_atoms_by_quality(fixed_cas, min_plddt=min_plddt, max_bfactor=max_bfactor)
    mobile_mask = filter_ca_atoms_by_quality(mobile_cas, min_plddt=min_plddt, max_bfactor=max_bfactor)

    # Combine with AND: both must pass
    return fixed_mask & mobile_mask
