"""Tests for sequence alignment operations."""

import gemmi
import numpy as np
import pytest

from pyprotalign.alignment import (
    _match_chain_centers,
    _match_chain_centers_hungarian,
    align_globally,
    align_hungarian,
    align_quaternary,
)
from pyprotalign.gemmi_utils import align_sequences, get_all_protein_chains


class TestAlignSequences:
    """Tests for align_sequences function."""

    def test_identical_sequences(self) -> None:
        """Test alignment of identical sequences."""
        seq1 = "ACDEFG"
        seq2 = "ACDEFG"
        pairs = align_sequences(seq1, seq2)

        expected = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))
        assert pairs == expected

    def test_single_mismatch(self) -> None:
        """Test alignment with single mismatch."""
        seq1 = "ACDEFG"
        seq2 = "ACDXFG"
        pairs = align_sequences(seq1, seq2)

        # Should align with mismatch at position 3
        expected = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))
        assert pairs == expected

    def test_gap_in_target(self) -> None:
        """Test alignment with gap in target sequence."""
        seq1 = "ACDEFGH"
        seq2 = "ACDXFG"
        pairs = align_sequences(seq1, seq2)

        # seq1 longer, should have gap in seq2 at end
        expected = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, None))
        assert pairs == expected

    def test_gap_in_query(self) -> None:
        """Test alignment with gap in query sequence."""
        seq1 = "ACDXFG"
        seq2 = "ACDEFGH"
        pairs = align_sequences(seq1, seq2)

        # seq2 longer, should have gap in seq1
        # Depending on alignment, could be at position 3 or end
        assert len(pairs) == 7
        # Check we have one gap in seq1
        gaps_in_seq1 = sum(1 for p1, p2 in pairs if p1 is None)
        assert gaps_in_seq1 == 1

    def test_very_different_sequences(self) -> None:
        """Test alignment of very different sequences."""
        seq1 = "AAA"
        seq2 = "GGG"
        pairs = align_sequences(seq1, seq2)

        # Should still align length 3, all mismatches
        expected = ((0, 0), (1, 1), (2, 2))
        assert pairs == expected

    def test_single_residue_each(self) -> None:
        """Test alignment of single residue sequences."""
        seq1 = "A"
        seq2 = "A"
        pairs = align_sequences(seq1, seq2)

        expected = ((0, 0),)
        assert pairs == expected

    def test_empty_alignment_length(self) -> None:
        """Test that all pairs account for both sequences."""
        seq1 = "ACDEFGHIK"
        seq2 = "ACDXFG"
        pairs = align_sequences(seq1, seq2)

        # Count non-gap positions
        seq1_positions = [p1 for p1, _ in pairs if p1 is not None]
        seq2_positions = [p2 for _, p2 in pairs if p2 is not None]

        # Should cover all positions in both sequences
        assert len(seq1_positions) == len(seq1)
        assert len(seq2_positions) == len(seq2)


class TestAlignMultiChain:
    """Tests for align_multi_chain function."""

    def _create_chain(self, name: str, sequence: str, missing_ca_indices: set[int] | None = None) -> gemmi.Chain:
        """Helper to create a chain with sequence.

        Args:
            name: Chain name
            sequence: Space-separated three-letter codes (e.g. "ALA GLY SER")
            missing_ca_indices: Residue indices to omit CA atoms for
        """
        chain = gemmi.Chain(name)
        missing = missing_ca_indices or set()
        residues = sequence.split()
        for i, res_name in enumerate(residues):
            res = gemmi.Residue()
            res.name = res_name  # Three-letter code
            res.seqid = gemmi.SeqId(str(i + 1))
            res.entity_type = gemmi.EntityType.Polymer
            if i not in missing:
                # Add CA atom
                atom = gemmi.Atom()
                atom.name = "CA"
                atom.element = gemmi.Element("C")
                atom.pos = gemmi.Position(float(i), 0.0, 0.0)
                res.add_atom(atom)
            chain.add_residue(res)
        return chain

    def _create_structure(self, chains: list[tuple[str, str, set[int] | None]]) -> gemmi.Structure:
        """Helper to create structure with multiple chains.

        Args:
            chains: List of (chain_name, sequence, missing_ca_indices) tuples
        """
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        for chain_name, sequence, missing_ca in chains:
            chain = self._create_chain(chain_name, sequence, missing_ca)
            model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()
        return structure

    def test_matching_chains(self) -> None:
        """Test alignment with matching chains."""
        # Create two structures with chains A and B
        fixed_st = self._create_structure([("A", "ALA GLY", None), ("B", "SER THR", None)])
        mobile_st = self._create_structure([("A", "ALA GLY", None), ("B", "SER THR", None)])

        # Extract ProteinChain objects
        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(fixed_map, mobile_map)

        # Should align 4 CA pairs (2 from A, 2 from B)
        assert num_aligned == 4
        assert chain_mapping == {"A": "A", "B": "B"}

    def test_different_chain_counts(self) -> None:
        """Test structures with different chain counts."""
        # Fixed has A, B, C; mobile has A, B
        fixed_st = self._create_structure([("A", "ALA GLY", None), ("B", "SER THR", None), ("C", "VAL LEU", None)])
        mobile_st = self._create_structure([("A", "ALA GLY", None), ("B", "SER THR", None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(fixed_map, mobile_map)

        # Should only use common chains A and B (4 pairs total)
        assert num_aligned == 4
        assert chain_mapping == {"A": "A", "B": "B"}

    def test_no_matching_chains(self) -> None:
        """Test error when no matching chains."""
        fixed_st = self._create_structure([("A", "ALA GLY", None)])
        mobile_st = self._create_structure([("B", "SER THR", None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        with pytest.raises(ValueError, match="No matching chain labels found"):
            align_globally(fixed_map, mobile_map)

    def test_insufficient_aligned_pairs(self) -> None:
        """Test error when fewer than 3 aligned pairs."""
        # Only 2 residues total
        fixed_st = self._create_structure([("A", "ALA GLY", None)])
        mobile_st = self._create_structure([("A", "ALA GLY", None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        with pytest.raises(ValueError, match="Need at least 3 aligned CA pairs"):
            align_globally(fixed_map, mobile_map)

    def test_coordinate_extraction(self) -> None:
        """Test that coordinates are correctly extracted."""
        fixed_st = self._create_structure([("A", "ALA GLY SER", None)])
        mobile_st = self._create_structure([("A", "ALA GLY SER", None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(fixed_map, mobile_map)

        # Should align 3 CA pairs
        assert num_aligned == 3
        # Check that rotation is identity (identical structures)
        import numpy as np

        assert np.allclose(rotation, np.eye(3), atol=1e-10)

    def test_chain_order_deterministic(self) -> None:
        """Test that chain order is deterministic (sorted)."""
        # Create with chains in different order
        fixed_st = self._create_structure([("C", "ALA GLY", None), ("A", "SER THR", None), ("B", "VAL LEU", None)])
        mobile_st = self._create_structure([("B", "VAL LEU", None), ("A", "SER THR", None), ("C", "ALA GLY", None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(fixed_map, mobile_map)

        # Chain mapping should be sorted (A, B, C)
        assert list(chain_mapping.keys()) == ["A", "B", "C"]

    def test_missing_ca_atoms_do_not_shift_indices(self) -> None:
        """Test alignment when CA atoms are missing in fixed or mobile."""
        fixed_st = self._create_structure([("A", "ALA GLY SER THR", {1})])
        mobile_st = self._create_structure([("A", "ALA GLY SER THR", None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(fixed_map, mobile_map)

        # Should align 3 CA pairs (index 1 missing in fixed)
        assert num_aligned == 3
        assert chain_mapping == {"A": "A"}


class TestAlignQuaternary:
    """Tests for align_quaternary function."""

    def _create_chain(
        self,
        name: str,
        sequence: str,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        missing_ca_indices: set[int] | None = None,
    ) -> gemmi.Chain:
        """Helper to create a chain with sequence and position offset.

        Args:
            name: Chain name
            sequence: Space-separated three-letter codes
            offset: XYZ offset to apply to all atoms
            missing_ca_indices: Residue indices to omit CA atoms for
        """
        chain = gemmi.Chain(name)
        missing = missing_ca_indices or set()
        residues = sequence.split()
        for i, res_name in enumerate(residues):
            res = gemmi.Residue()
            res.name = res_name
            res.seqid = gemmi.SeqId(str(i + 1))
            res.entity_type = gemmi.EntityType.Polymer
            if i not in missing:
                atom = gemmi.Atom()
                atom.name = "CA"
                atom.element = gemmi.Element("C")
                atom.pos = gemmi.Position(float(i) + offset[0], offset[1], offset[2])
                res.add_atom(atom)
            chain.add_residue(res)
        return chain

    def _create_structure(
        self, chains: list[tuple[str, str, tuple[float, float, float], set[int] | None]]
    ) -> gemmi.Structure:
        """Helper to create structure with multiple chains.

        Args:
            chains: List of (chain_name, sequence, offset, missing_ca_indices) tuples
        """
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        for chain_name, sequence, offset, missing_ca in chains:
            chain = self._create_chain(chain_name, sequence, offset, missing_ca)
            model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()
        return structure

    def test_perfect_match_same_labels(self) -> None:
        """Test quaternary alignment with matching labels and positions."""
        # Two chains, same labels, offset in Y
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_quaternary(
            fixed_chains, mobile_chains, distance_threshold=15.0
        )

        # Should match both chains
        assert len(chain_mapping) == 2
        assert chain_mapping == {"A": "A", "B": "B"}
        assert num_aligned == 6  # 3 residues × 2 chains

    def test_permuted_labels(self) -> None:
        """Test quaternary alignment with swapped chain labels."""
        # Fixed: A at (0,0,0), B at (0,10,0)
        # Mobile: C at (0,0,0), D at (0,10,0)
        # Should match A->C (seed), B->D (by proximity)
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("C", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("D", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_quaternary(
            fixed_chains, mobile_chains, distance_threshold=15.0
        )

        # Should match A->C (seed), B->D (proximity)
        assert len(chain_mapping) == 2
        assert chain_mapping == {"A": "C", "B": "D"}

    def test_missing_chain_in_mobile(self) -> None:
        """Test when fixed has more chains than mobile."""
        # Fixed has 3 chains, mobile has 2
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
                ("C", "ILE MET PHE", (0.0, 20.0, 0.0), None),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_quaternary(
            fixed_chains, mobile_chains, distance_threshold=15.0
        )

        # Should match A->A, B->B only (C has no match)
        assert len(chain_mapping) == 2
        assert chain_mapping == {"A": "A", "B": "B"}

    def test_distance_threshold_filters(self) -> None:
        """Test that distance threshold filters out distant chains."""
        # Fixed: A at (0,0,0), B at (0,10,0)
        # Mobile: A at (0,0,0), C at (0,-10,0) - C is far from B after seed alignment
        # After aligning seed A->A, C will be at (0,-10,0) but B is at (0,10,0)
        # Distance between centers ~20 Angstroms
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("C", "THR VAL LEU", (0.0, -10.0, 0.0), None),  # Far from B
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_quaternary(
            fixed_chains,
            mobile_chains,
            distance_threshold=5.0,  # Small threshold
        )

        # Should only match seed chain A (C is too far from B)
        assert len(chain_mapping) == 1
        assert chain_mapping == {"A": "A"}


class TestAlignHungarian:
    """Tests for align_hungarian function and Hungarian chain matching helper."""

    def _create_chain(
        self,
        name: str,
        sequence: str,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        missing_ca_indices: set[int] | None = None,
    ) -> gemmi.Chain:
        chain = gemmi.Chain(name)
        missing = missing_ca_indices or set()
        residues = sequence.split()
        for i, res_name in enumerate(residues):
            res = gemmi.Residue()
            res.name = res_name
            res.seqid = gemmi.SeqId(str(i + 1))
            res.entity_type = gemmi.EntityType.Polymer
            if i not in missing:
                atom = gemmi.Atom()
                atom.name = "CA"
                atom.element = gemmi.Element("C")
                atom.pos = gemmi.Position(float(i) + offset[0], offset[1], offset[2])
                res.add_atom(atom)
            chain.add_residue(res)
        return chain

    def _create_structure(
        self, chains: list[tuple[str, str, tuple[float, float, float], set[int] | None]]
    ) -> gemmi.Structure:
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        for chain_name, sequence, offset, missing_ca in chains:
            chain = self._create_chain(chain_name, sequence, offset, missing_ca)
            model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()
        return structure

    def test_permuted_labels(self) -> None:
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("C", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("D", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_hungarian(
            fixed_chains, mobile_chains, distance_threshold=15.0
        )

        assert chain_mapping == {"A": "C", "B": "D"}
        assert num_aligned == 6
        assert rmsd < 1e-6

    def test_threshold_filters(self) -> None:
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), None),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("C", "THR VAL LEU", (0.0, -10.0, 0.0), None),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_hungarian(
            fixed_chains,
            mobile_chains,
            distance_threshold=5.0,
        )

        assert chain_mapping == {"A": "A"}
        assert num_aligned == 3
        assert rmsd < 1e-6

    def test_invalid_seed_chain(self) -> None:
        fixed_st = self._create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0), None)])
        mobile_st = self._create_structure([("B", "ALA GLY SER", (0.0, 0.0, 0.0), None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        with pytest.raises(ValueError, match="Fixed seed chain 'Z' not found"):
            align_hungarian(fixed_chains, mobile_chains, fixed_seed="Z")

    def test_hungarian_finds_better_mapping_than_greedy(self) -> None:
        distances = np.array(
            [
                [1.0, 2.0],
                [1.1, 100.0],
            ],
            dtype=float,
        )

        greedy_pairs = _match_chain_centers(distances, distance_threshold=200.0)
        hungarian_pairs = _match_chain_centers_hungarian(distances, distance_threshold=200.0)

        assert greedy_pairs == [(0, 0), (1, 1)]
        assert hungarian_pairs == [(0, 1), (1, 0)]

        greedy_cost = float(sum(distances[i, j] for i, j in greedy_pairs))
        hungarian_cost = float(sum(distances[i, j] for i, j in hungarian_pairs))
        assert hungarian_cost < greedy_cost

    def test_hungarian_no_matches_with_small_threshold(self) -> None:
        distances = np.array([[10.0, 12.0], [11.0, 13.0]], dtype=float)
        with pytest.raises(ValueError, match="No matching chains found"):
            _match_chain_centers_hungarian(distances, distance_threshold=5.0)

    def test_insufficient_pairs_error(self) -> None:
        """Test error when Hungarian seed chains have too few aligned pairs."""
        fixed_st = self._create_structure([("A", "ALA GLY", (0.0, 0.0, 0.0), None)])
        mobile_st = self._create_structure([("A", "ALA GLY", (0.0, 0.0, 0.0), None)])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        with pytest.raises(ValueError, match="Need at least 3 aligned CA pairs"):
            align_hungarian(fixed_chains, mobile_chains, distance_threshold=10.0)

    def test_seed_chain_selection(self) -> None:
        """Test specifying Hungarian seed chains explicitly."""
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU ILE", (0.0, 10.0, 0.0), None),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("X", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("Y", "THR VAL LEU ILE", (0.0, 10.0, 0.0), None),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_hungarian(
            fixed_chains, mobile_chains, distance_threshold=15.0, fixed_seed="B", mobile_seed="Y"
        )

        # Should use B->Y as seed
        assert chain_mapping == {"B": "Y", "A": "X"}
        assert len(chain_mapping) == 2

    def test_quaternary_skips_chains_without_ca(self) -> None:
        """Test that Hungarian matching skips chains without CA atoms."""
        fixed_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("B", "THR VAL LEU", (0.0, 10.0, 0.0), {0, 1, 2}),
            ]
        )
        mobile_st = self._create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0), None),
                ("C", "THR VAL LEU", (0.0, 10.0, 0.0), {0, 1, 2}),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned, chain_mapping = align_hungarian(
            fixed_chains, mobile_chains, distance_threshold=15.0
        )

        assert len(chain_mapping) == 1
        assert chain_mapping == {"A": "A"}
