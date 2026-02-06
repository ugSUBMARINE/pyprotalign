"""Internal representation of protein chain for alignment.

This module provides a library-agnostic representation of protein chains
that can be constructed from gemmi or Biopython objects.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        self.coords = np.asarray(self.coords, dtype=float)
        self.b_factors = np.asarray(self.b_factors, dtype=float)
        self.occupancies = np.asarray(self.occupancies, dtype=float)

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

    def get_bfac_occ_mask(
        self,
        min_bfactor: float = -np.inf,
        max_bfactor: float = np.inf,
        min_occ: float = -np.inf,
        idx_list: list[int] | None = None,
    ) -> NDArray[np.bool_]:
        """Get mask of residues passing B-factor and occupancy thresholds.

        Args:
            min_bfactor: Minimum B-factor threshold (inclusive)
            max_bfactor: Maximum B-factor threshold (inclusive)
            min_occ: Minimum occupancy threshold (inclusive)
            idx_list: Optional list of residue indices to consider (if None, consider all residues)
        Returns:
            Boolean mask array (True = pass thresholds, False = fail)
        """
        if idx_list is None:
            bfactor_mask = (self.b_factors >= min_bfactor) & (self.b_factors <= max_bfactor)
            occ_mask = self.occupancies >= min_occ
        else:
            tmp_bfactor = self.b_factors[idx_list]
            bfactor_mask = (tmp_bfactor >= min_bfactor) & (tmp_bfactor <= max_bfactor)
            occ_mask = self.occupancies[idx_list] >= min_occ
        return bfactor_mask & occ_mask
