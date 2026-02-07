"""Tests for Biopython-backed utilities."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("Bio")

from Bio.PDB import PDBParser  # type: ignore[import-not-found]

from pyprotalign.bio_utils import align_sequences, create_chain, load_structure


class TestBioAlignSequences:
    """Tests for Biopython align_sequences."""

    def test_identical_sequences(self) -> None:
        """Test alignment of identical sequences."""
        seq1 = "ACDEFG"
        seq2 = "ACDEFG"
        pairs = align_sequences(seq1, seq2)
        expected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        assert pairs == expected

    def test_gap_coverage(self) -> None:
        """Test alignment covers all positions with gaps."""
        seq1 = "ACDEFGHIK"
        seq2 = "ACDXFG"
        pairs = align_sequences(seq1, seq2)

        seq1_positions = [p1 for p1, _ in pairs if p1 is not None]
        seq2_positions = [p2 for _, p2 in pairs if p2 is not None]

        assert len(seq1_positions) == len(seq1)
        assert len(seq2_positions) == len(seq2)


class TestBioCreateChain:
    """Tests for Biopython create_chain."""

    def test_create_chain_basic(self, tmp_path: Path) -> None:
        """Test creating ProteinChain from Biopython Chain."""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 11.00           C\n"
            "ATOM      3  C   ALA A   1       2.100   1.400   0.000  1.00 12.00           C\n"
            "ATOM      4  N   CYS A   2       3.000   1.500   0.000  1.00 13.00           N\n"
            "ATOM      5  CA  CYS A   2       4.000   1.500   0.000  0.80 14.00           C\n"
            "ATOM      6  C   CYS A   2       5.000   2.000   0.000  1.00 15.00           C\n"
            "END\n"
        )

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(pdb_path))
        chain = structure[0]["A"]

        pchain = create_chain(chain)

        assert pchain.chain_id == "A"
        assert pchain.sequence == "AC"
        assert pchain.coords.shape == (2, 3)
        np.testing.assert_allclose(pchain.coords[0], [1.458, 0.0, 0.0])
        np.testing.assert_allclose(pchain.coords[1], [4.0, 1.5, 0.0])
        np.testing.assert_allclose(pchain.b_factors, [11.0, 14.0])
        np.testing.assert_allclose(pchain.occupancies, [1.0, 0.8])


class TestBioLoadStructure:
    """Tests for Biopython load_structure."""

    def test_load_pdb_file(self, tmp_path: Path) -> None:
        """Test loading a PDB file."""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N\n"
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 10.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.000   0.000  1.00 10.00           C\n"
            "END\n"
        )

        structure = load_structure(str(pdb_path))
        assert len(structure) == 1
        assert len(structure[0]) > 0
