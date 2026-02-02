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
