"""Utility functions for working with Biopython library in pyprotalign package."""

from __future__ import annotations

try:
    from Bio.PDB.Polypeptide import PPBuilder, is_aa  # type: ignore[import-not-found]
    from Bio.SeqUtils import seq1  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError("Biopython is required. Install with: pip install biopython") from e

import numpy as np

from .chain import ProteinChain


def create_chain(chain: object, chain_id: str | None = None) -> ProteinChain:
    """Create ProteinChain from Biopython Chain object.

    Args:
        chain: Biopython Chain object (Bio.PDB.Chain.Chain)
        chain_id: Optional chain ID override (uses chain.id if None)

    Returns:
        ProteinChain instance

    Raises:
        ValueError: If chain has no residues
        ImportError: If Biopython is not installed
    """
    # Get chain ID
    if chain_id is None:
        chain_id = chain.id  # type: ignore[attr-defined]

    # Build polypeptides (handles missing residues)
    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(chain)

    if len(polypeptides) == 0:
        raise ValueError(f"Chain '{chain_id}' has no polypeptide segments")

    # Collect residues from all polypeptide segments
    residues = []
    for pp in polypeptides:
        residues.extend(pp)

    if len(residues) == 0:
        raise ValueError(f"Chain '{chain_id}' has no amino acid residues")

    # Extract sequence and data
    sequence_chars = []
    coords_list = []
    b_factors_list = []
    occupancies_list = []

    for residue in residues:
        # Get single-letter code
        if is_aa(residue):
            sequence_chars.append(residue.get_resname())
        else:
            # Skip non-standard residues
            continue

        # Extract CA atom if present
        if "CA" in residue:
            ca_atom = residue["CA"]
            coords_list.append(ca_atom.coord)
            b_factors_list.append(ca_atom.bfactor)
            occupancies_list.append(ca_atom.occupancy)
        else:
            # Missing CA atom
            coords_list.append([np.nan, np.nan, np.nan])
            b_factors_list.append(np.nan)
            occupancies_list.append(np.nan)

    # Convert 3-letter codes to 1-letter
    sequence = seq1("".join(sequence_chars))

    coords = np.array(coords_list, dtype=float)
    b_factors = np.array(b_factors_list, dtype=float)
    occupancies = np.array(occupancies_list, dtype=float)

    return ProteinChain(
        chain_id=chain_id,
        sequence=sequence,
        coords=coords,
        b_factors=b_factors,
        occupancies=occupancies,
    )
