"""Internal representation of protein chain for alignment.

This module provides a library-agnostic representation of protein chains
that can be constructed from gemmi or Biopython objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np
from numpy.typing import NDArray


@dataclass
class ProteinChain:
    """Internal representation of protein chain for structure alignment.

    Attributes:
        chain_id: Chain identifier (e.g. "A", "B")
        sequence: Single-letter amino acid sequence
        coords: Atom coordinates (N, 3) array, aligned to sequence indices
        b_factors: B-factor or pLDDT values per residue (N,) array
        occupancies: Occupancy values per residue (N,) array

    Notes:
        - All arrays must have same length as sequence
        - Missing CA atoms represented as NaN in coords
        - Missing values in b_factors/occupancies represented as NaN
    """

    chain_id: str
    sequence: str
    coords: NDArray[np.floating]
    b_factors: NDArray[np.floating]
    occupancies: NDArray[np.floating]

    def __post_init__(self) -> None:
        """Validate ProteinChain data after initialization.

        Raises:
            ValueError: If validation fails
        """
        # Convert to numpy arrays with float64 dtype
        self.coords = np.asarray(self.coords, dtype=np.float64)
        self.b_factors = np.asarray(self.b_factors, dtype=np.float64)
        self.occupancies = np.asarray(self.occupancies, dtype=np.float64)

        n_residues = len(self.sequence)

        # Validate sequence
        if n_residues == 0:
            raise ValueError("Sequence cannot be empty")

        # Validate coords shape
        if self.coords.ndim != 2:
            raise ValueError(f"coords must be 2D array, got shape {self.coords.shape}")
        if self.coords.shape != (n_residues, 3):
            raise ValueError(f"coords shape {self.coords.shape} does not match sequence length {n_residues}")

        # Validate b_factors shape
        if self.b_factors.ndim != 1:
            raise ValueError(f"b_factors must be 1D array, got shape {self.b_factors.shape}")
        if self.b_factors.shape[0] != n_residues:
            raise ValueError(f"b_factors length {self.b_factors.shape[0]} does not match sequence length {n_residues}")

        # Validate occupancies shape
        if self.occupancies.ndim != 1:
            raise ValueError(f"occupancies must be 1D array, got shape {self.occupancies.shape}")
        if self.occupancies.shape[0] != n_residues:
            raise ValueError(
                f"occupancies length {self.occupancies.shape[0]} does not match sequence length {n_residues}"
            )

        # Validate chain_id
        if not self.chain_id:
            raise ValueError("chain_id cannot be empty")

    @classmethod
    def from_gemmi(cls, chain: gemmi.Chain) -> ProteinChain:
        """Create ProteinChain from gemmi.Chain object.

        Args:
            chain: gemmi Chain object (must be a protein chain)

        Returns:
            ProteinChain instance

        Raises:
            ValueError: If chain is not a protein chain or has no residues
        """
        # Get polymer residues
        polymer = chain.get_polymer()
        if len(polymer) == 0 or polymer.check_polymer_type() != gemmi.PolymerType.PeptideL:
            raise ValueError(f"Chain '{chain.name}' is not a protein chain")

        # Extract one-letter sequence, removing gaps that are placed at chain breaks
        sequence = polymer.make_one_letter_sequence().replace("-", "")
        n_residues = len(sequence)

        # Initialize arrays with NaN (for missing atoms)
        coords = np.full((n_residues, 3), np.nan, dtype=np.float64)
        b_factors = np.full(n_residues, np.nan, dtype=np.float64)
        occupancies = np.full(n_residues, np.nan, dtype=np.float64)

        # Extract CA atom data
        for i, residue in enumerate(polymer):
            ca = residue.get_ca()
            if ca:
                coords[i] = [ca.pos.x, ca.pos.y, ca.pos.z]
                b_factors[i] = ca.b_iso
                occupancies[i] = ca.occ

        return cls(
            chain_id=chain.name,
            sequence=sequence,
            coords=coords,
            b_factors=b_factors,
            occupancies=occupancies,
        )

    @classmethod
    def from_biopython(cls, chain: object, chain_id: str | None = None) -> ProteinChain:
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
        try:
            from Bio.PDB.Polypeptide import PPBuilder, is_aa  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("Biopython is required for from_biopython(). Install with: pip install biopython") from e

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
        from Bio.SeqUtils import seq1  # type: ignore[import-not-found]

        sequence = seq1("".join(sequence_chars))

        coords = np.array(coords_list, dtype=np.float64)
        b_factors = np.array(b_factors_list, dtype=np.float64)
        occupancies = np.array(occupancies_list, dtype=np.float64)

        return cls(
            chain_id=chain_id,
            sequence=sequence,
            coords=coords,
            b_factors=b_factors,
            occupancies=occupancies,
        )

    def get_aligned_indices(self) -> NDArray[np.intp]:
        """Get indices of residues with CA atoms present (non-NaN coordinates).

        Returns:
            Array of indices where CA atoms are present
        """
        # Check first coordinate (x) for NaN - if x is NaN, all coords are NaN
        return np.where(~np.isnan(self.coords[:, 0]))[0]

    def get_ca_atoms(self) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Get CA atom data for residues with CA atoms present.

        Returns:
            Tuple of (coords, b_factors, occupancies) with NaN rows filtered out
        """
        mask = ~np.isnan(self.coords[:, 0])
        return (
            self.coords[mask],
            self.b_factors[mask],
            self.occupancies[mask],
        )

    def filter_by_quality(
        self,
        min_plddt: float | None = None,
        max_bfactor: float | None = None,
    ) -> NDArray[np.bool_]:
        """Filter residues by B-factor or pLDDT quality metric.

        Args:
            min_plddt: Minimum pLDDT threshold (residues with b_factor < threshold filtered out)
            max_bfactor: Maximum B-factor threshold (residues with b_factor > threshold filtered out)

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

        # Initialize mask to False (filter out) for all residues
        mask = np.zeros(len(self.sequence), dtype=bool)

        # Only keep residues with CA atoms and passing quality threshold
        has_ca = ~np.isnan(self.coords[:, 0])
        has_bfactor = ~np.isnan(self.b_factors)

        if min_plddt is not None:
            # pLDDT filter: keep if >= threshold
            mask = has_ca & has_bfactor & (self.b_factors >= min_plddt)
        elif max_bfactor is not None:
            # B-factor filter: keep if <= threshold
            mask = has_ca & has_bfactor & (self.b_factors <= max_bfactor)

        return mask
