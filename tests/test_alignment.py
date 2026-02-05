"""Tests for sequence alignment operations."""

import gemmi
import pytest

from pyprotalign.alignment import align_multi_chain, align_quaternary, align_sequences


class TestAlignSequences:
    """Tests for align_sequences function."""

    def test_identical_sequences(self) -> None:
        """Test alignment of identical sequences."""
        seq1 = "ACDEFG"
        seq2 = "ACDEFG"
        pairs = align_sequences(seq1, seq2)

        expected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        assert pairs == expected

    def test_single_mismatch(self) -> None:
        """Test alignment with single mismatch."""
        seq1 = "ACDEFG"
        seq2 = "ACDXFG"
        pairs = align_sequences(seq1, seq2)

        # Should align with mismatch at position 3
        expected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        assert pairs == expected

    def test_gap_in_target(self) -> None:
        """Test alignment with gap in target sequence."""
        seq1 = "ACDEFGH"
        seq2 = "ACDXFG"
        pairs = align_sequences(seq1, seq2)

        # seq1 longer, should have gap in seq2 at end
        expected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, None)]
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
        expected = [(0, 0), (1, 1), (2, 2)]
        assert pairs == expected

    def test_single_residue_each(self) -> None:
        """Test alignment of single residue sequences."""
        seq1 = "A"
        seq2 = "A"
        pairs = align_sequences(seq1, seq2)

        expected = [(0, 0)]
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

        fixed_coords, mobile_coords, chain_ids = align_multi_chain(fixed_st, mobile_st)

        # Should have 4 pairs (2 from A, 2 from B)
        assert len(fixed_coords) == 4
        assert len(mobile_coords) == 4
        assert len(chain_ids) == 4
        assert chain_ids == ["A", "A", "B", "B"]

    def test_different_chain_counts(self) -> None:
        """Test structures with different chain counts."""
        # Fixed has A, B, C; mobile has A, B
        fixed_st = self._create_structure([("A", "ALA GLY", None), ("B", "SER THR", None), ("C", "VAL LEU", None)])
        mobile_st = self._create_structure([("A", "ALA GLY", None), ("B", "SER THR", None)])

        fixed_coords, mobile_coords, chain_ids = align_multi_chain(fixed_st, mobile_st)

        # Should only use common chains A and B (4 pairs total)
        assert len(fixed_coords) == 4
        assert chain_ids == ["A", "A", "B", "B"]

    def test_no_matching_chains(self) -> None:
        """Test error when no matching chains."""
        fixed_st = self._create_structure([("A", "ALA GLY", None)])
        mobile_st = self._create_structure([("B", "SER THR", None)])

        with pytest.raises(ValueError, match="No matching protein chains found"):
            align_multi_chain(fixed_st, mobile_st)

    def test_insufficient_aligned_pairs(self) -> None:
        """Test error when fewer than 3 aligned pairs."""
        # Only 2 residues total
        fixed_st = self._create_structure([("A", "ALA GLY", None)])
        mobile_st = self._create_structure([("A", "ALA GLY", None)])

        with pytest.raises(ValueError, match="Need at least 3 aligned CA pairs"):
            align_multi_chain(fixed_st, mobile_st)

    def test_coordinate_extraction(self) -> None:
        """Test that coordinates are correctly extracted."""
        fixed_st = self._create_structure([("A", "ALA GLY SER", None)])
        mobile_st = self._create_structure([("A", "ALA GLY SER", None)])

        fixed_coords, mobile_coords, chain_ids = align_multi_chain(fixed_st, mobile_st)

        # Check shapes
        assert fixed_coords.shape == (3, 3)
        assert mobile_coords.shape == (3, 3)

        # Fixed coords should be (0,0,0), (1,0,0), (2,0,0)
        assert fixed_coords[0, 0] == 0.0
        assert fixed_coords[1, 0] == 1.0
        assert fixed_coords[2, 0] == 2.0

    def test_chain_order_deterministic(self) -> None:
        """Test that chain order is deterministic (sorted)."""
        # Create with chains in different order
        fixed_st = self._create_structure([("C", "ALA GLY", None), ("A", "SER THR", None), ("B", "VAL LEU", None)])
        mobile_st = self._create_structure([("B", "VAL LEU", None), ("A", "SER THR", None), ("C", "ALA GLY", None)])

        _, _, chain_ids = align_multi_chain(fixed_st, mobile_st)

        # Should be in sorted order: A, A, B, B, C, C
        assert chain_ids == ["A", "A", "B", "B", "C", "C"]

    def test_missing_ca_atoms_do_not_shift_indices(self) -> None:
        """Test alignment when CA atoms are missing in fixed or mobile."""
        fixed_st = self._create_structure([("A", "ALA GLY SER THR", {1})])
        mobile_st = self._create_structure([("A", "ALA GLY SER THR", None)])

        fixed_coords, mobile_coords, chain_ids = align_multi_chain(fixed_st, mobile_st)

        assert len(fixed_coords) == 3
        assert len(mobile_coords) == 3
        assert chain_ids == ["A", "A", "A"]
        assert [float(x) for x in fixed_coords[:, 0]] == [0.0, 2.0, 3.0]
        assert [float(x) for x in mobile_coords[:, 0]] == [0.0, 2.0, 3.0]


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

        fixed_coords, mobile_coords, chain_pairs = align_quaternary(fixed_st, mobile_st, distance_threshold=15.0)

        # Should match both chains
        assert len(chain_pairs) == 2
        assert ("A", "A") in chain_pairs
        assert ("B", "B") in chain_pairs
        assert len(fixed_coords) == 6  # 3 residues × 2 chains

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

        fixed_coords, mobile_coords, chain_pairs = align_quaternary(fixed_st, mobile_st, distance_threshold=15.0)

        # Should match A->C (seed), B->D (proximity)
        assert len(chain_pairs) == 2
        assert ("A", "C") in chain_pairs
        assert ("B", "D") in chain_pairs

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

        fixed_coords, mobile_coords, chain_pairs = align_quaternary(fixed_st, mobile_st, distance_threshold=15.0)

        # Should match A->A, B->B only (C has no match)
        assert len(chain_pairs) == 2
        assert ("A", "A") in chain_pairs
        assert ("B", "B") in chain_pairs

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

        fixed_coords, mobile_coords, chain_pairs = align_quaternary(
            fixed_st,
            mobile_st,
            distance_threshold=5.0,  # Small threshold
        )

        # Should only match seed chain A (C is too far from B)
        assert len(chain_pairs) == 1
        assert ("A", "A") in chain_pairs

    def test_insufficient_pairs_error(self) -> None:
        """Test error when seed chains have too few aligned pairs."""
        # Only 2 residues in seed - not enough
        fixed_st = self._create_structure([("A", "ALA GLY", (0.0, 0.0, 0.0), None)])
        mobile_st = self._create_structure([("A", "ALA GLY", (0.0, 0.0, 0.0), None)])

        with pytest.raises(ValueError, match="Need at least 3 aligned CA pairs in seed chains"):
            align_quaternary(fixed_st, mobile_st, distance_threshold=10.0)

    def test_seed_chain_selection(self) -> None:
        """Test specifying seed chains explicitly."""
        # Multiple chains, specify B as seed
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

        fixed_coords, mobile_coords, chain_pairs = align_quaternary(
            fixed_st, mobile_st, distance_threshold=15.0, seed_fixed_chain="B", seed_mobile_chain="Y"
        )

        # Should use B->Y as seed
        assert ("B", "Y") in chain_pairs
        assert len(chain_pairs) == 2

    def test_quaternary_skips_chains_without_ca(self) -> None:
        """Test that chains without CA atoms are skipped in matching."""
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

        _, _, chain_pairs = align_quaternary(fixed_st, mobile_st, distance_threshold=15.0)

        assert len(chain_pairs) == 1
        assert ("A", "A") in chain_pairs
