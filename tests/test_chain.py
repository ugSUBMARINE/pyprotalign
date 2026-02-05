"""Tests for ProteinChain dataclass."""

from pathlib import Path

import gemmi
import numpy as np
import pytest

from pyprotalign.chain import ProteinChain


class TestProteinChainValidation:
    """Tests for ProteinChain validation logic."""

    def test_valid_chain(self) -> None:
        """Test creating valid ProteinChain."""
        chain = ProteinChain(
            chain_id="A",
            sequence="ACDEF",
            coords=np.random.rand(5, 3),
            b_factors=np.random.rand(5),
            occupancies=np.ones(5),
        )
        assert chain.chain_id == "A"
        assert len(chain.sequence) == 5
        assert chain.coords.shape == (5, 3)

    def test_empty_sequence_raises(self) -> None:
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            ProteinChain(
                chain_id="A",
                sequence="",
                coords=np.empty((0, 3)),
                b_factors=np.empty(0),
                occupancies=np.empty(0),
            )

    def test_mismatched_coords_length_raises(self) -> None:
        """Test that mismatched coords length raises ValueError."""
        with pytest.raises(ValueError, match="does not match sequence length"):
            ProteinChain(
                chain_id="A",
                sequence="ACDEF",
                coords=np.random.rand(3, 3),  # Wrong length
                b_factors=np.random.rand(5),
                occupancies=np.ones(5),
            )

    def test_wrong_coords_dimensions_raises(self) -> None:
        """Test that wrong coords dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be 2D array"):
            ProteinChain(
                chain_id="A",
                sequence="AC",
                coords=np.random.rand(2),  # 1D instead of 2D
                b_factors=np.random.rand(2),
                occupancies=np.ones(2),
            )

    def test_mismatched_bfactors_length_raises(self) -> None:
        """Test that mismatched b_factors length raises ValueError."""
        with pytest.raises(ValueError, match="does not match sequence length"):
            ProteinChain(
                chain_id="A",
                sequence="ACDEF",
                coords=np.random.rand(5, 3),
                b_factors=np.random.rand(3),  # Wrong length
                occupancies=np.ones(5),
            )

    def test_mismatched_occupancies_length_raises(self) -> None:
        """Test that mismatched occupancies length raises ValueError."""
        with pytest.raises(ValueError, match="does not match sequence length"):
            ProteinChain(
                chain_id="A",
                sequence="ACDEF",
                coords=np.random.rand(5, 3),
                b_factors=np.random.rand(5),
                occupancies=np.ones(3),  # Wrong length
            )

    def test_empty_chain_id_raises(self) -> None:
        """Test that empty chain_id raises ValueError."""
        with pytest.raises(ValueError, match="chain_id cannot be empty"):
            ProteinChain(
                chain_id="",
                sequence="AC",
                coords=np.random.rand(2, 3),
                b_factors=np.random.rand(2),
                occupancies=np.ones(2),
            )


class TestFromGemmi:
    """Tests for ProteinChain.from_gemmi()."""

    def test_from_gemmi_basic(self, tmp_path: Path) -> None:
        """Test creating ProteinChain from gemmi.Chain."""
        # Create test structure
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.50           C\n"
            "ATOM      2  CA  CYS A   2       4.000   5.000   6.000  0.80 20.30           C\n"
            "END\n"
        )

        structure = gemmi.read_structure(str(pdb_path))
        structure.setup_entities()
        chain = structure[0]["A"]

        pchain = ProteinChain.from_gemmi(chain)

        assert pchain.chain_id == "A"
        assert pchain.sequence == "AC"
        assert pchain.coords.shape == (2, 3)
        np.testing.assert_allclose(pchain.coords[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(pchain.coords[1], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(pchain.b_factors, [10.5, 20.3])
        np.testing.assert_allclose(pchain.occupancies, [1.0, 0.8])

    def test_from_gemmi_missing_ca(self, tmp_path: Path) -> None:
        """Test handling missing CA atoms."""
        # Create structure with missing CA for second residue
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.50           C\n"
            "ATOM      2  N   CYS A   2       4.000   5.000   6.000  1.00 20.30           N\n"
            "END\n"
        )

        structure = gemmi.read_structure(str(pdb_path))
        structure.setup_entities()
        chain = structure[0]["A"]

        pchain = ProteinChain.from_gemmi(chain)

        assert pchain.sequence == "AC"
        assert not np.isnan(pchain.coords[0]).any()
        assert np.isnan(pchain.coords[1]).all()
        assert not np.isnan(pchain.b_factors[0])
        assert np.isnan(pchain.b_factors[1])

    def test_from_gemmi_non_protein_raises(self) -> None:
        """Test that non-protein chain raises ValueError."""
        structure = gemmi.Structure()
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        # Empty chain (no residues)
        model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()

        with pytest.raises(ValueError, match="is not a protein chain"):
            ProteinChain.from_gemmi(structure[0]["A"])


class TestFromBiopython:
    """Tests for ProteinChain.from_biopython()."""

    def test_from_biopython_not_installed(self) -> None:
        """Test error when Biopython not available."""

        # Create a mock chain object
        class MockChain:
            id = "A"

        with pytest.raises(ImportError, match="Biopython is required"):
            ProteinChain.from_biopython(MockChain())


class TestGetAlignedIndices:
    """Tests for get_aligned_indices method."""

    def test_get_aligned_indices_all_present(self) -> None:
        """Test getting indices when all CA atoms present."""
        chain = ProteinChain(
            chain_id="A",
            sequence="ACDEF",
            coords=np.random.rand(5, 3),
            b_factors=np.random.rand(5),
            occupancies=np.ones(5),
        )
        indices = chain.get_aligned_indices()
        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])

    def test_get_aligned_indices_some_missing(self) -> None:
        """Test getting indices with missing CA atoms."""
        coords = np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],
                [4.0, 5.0, 6.0],
                [np.nan, np.nan, np.nan],
                [7.0, 8.0, 9.0],
            ]
        )
        chain = ProteinChain(
            chain_id="A",
            sequence="ACDEF",
            coords=coords,
            b_factors=np.random.rand(5),
            occupancies=np.ones(5),
        )
        indices = chain.get_aligned_indices()
        np.testing.assert_array_equal(indices, [0, 2, 4])


class TestGetCaAtoms:
    """Tests for get_ca_atoms method."""

    def test_get_ca_atoms_filters_nan(self) -> None:
        """Test that get_ca_atoms filters out NaN entries."""
        coords = np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],
                [4.0, 5.0, 6.0],
            ]
        )
        b_factors = np.array([10.0, 20.0, 30.0])
        occupancies = np.array([1.0, 0.5, 0.8])

        chain = ProteinChain(
            chain_id="A",
            sequence="ACD",
            coords=coords,
            b_factors=b_factors,
            occupancies=occupancies,
        )

        coords, bfacts, occs = chain.get_ca_atoms()

        assert coords.shape == (2, 3)
        np.testing.assert_allclose(coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(bfacts, [10.0, 30.0])
        np.testing.assert_allclose(occs, [1.0, 0.8])


class TestFilterByQuality:
    """Tests for filter_by_quality method."""

    def test_filter_by_plddt(self) -> None:
        """Test filtering by minimum pLDDT threshold."""
        chain = ProteinChain(
            chain_id="A",
            sequence="ACDEF",
            coords=np.random.rand(5, 3),
            b_factors=np.array([90.0, 50.0, 80.0, 30.0, 70.0]),
            occupancies=np.ones(5),
        )

        mask = chain.filter_by_quality(min_plddt=70.0)
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_filter_by_bfactor(self) -> None:
        """Test filtering by maximum B-factor threshold."""
        chain = ProteinChain(
            chain_id="A",
            sequence="ACDEF",
            coords=np.random.rand(5, 3),
            b_factors=np.array([10.0, 50.0, 20.0, 80.0, 30.0]),
            occupancies=np.ones(5),
        )

        mask = chain.filter_by_quality(max_bfactor=30.0)
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_filter_excludes_missing_ca(self) -> None:
        """Test that filter excludes residues with missing CA atoms."""
        coords = np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],  # Missing CA
                [4.0, 5.0, 6.0],
            ]
        )
        chain = ProteinChain(
            chain_id="A",
            sequence="ACD",
            coords=coords,
            b_factors=np.array([80.0, 90.0, 85.0]),
            occupancies=np.ones(3),
        )

        mask = chain.filter_by_quality(min_plddt=70.0)
        expected = np.array([True, False, True])  # Middle one excluded (no CA)
        np.testing.assert_array_equal(mask, expected)

    def test_filter_excludes_missing_bfactor(self) -> None:
        """Test that filter excludes residues with missing B-factor."""
        chain = ProteinChain(
            chain_id="A",
            sequence="ACD",
            coords=np.random.rand(3, 3),
            b_factors=np.array([80.0, np.nan, 85.0]),
            occupancies=np.ones(3),
        )

        mask = chain.filter_by_quality(min_plddt=70.0)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_filter_mutual_exclusivity_raises(self) -> None:
        """Test that specifying both filters raises ValueError."""
        chain = ProteinChain(
            chain_id="A",
            sequence="AC",
            coords=np.random.rand(2, 3),
            b_factors=np.array([80.0, 90.0]),
            occupancies=np.ones(2),
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            chain.filter_by_quality(min_plddt=70.0, max_bfactor=30.0)

    def test_filter_no_params_raises(self) -> None:
        """Test that specifying neither filter raises ValueError."""
        chain = ProteinChain(
            chain_id="A",
            sequence="AC",
            coords=np.random.rand(2, 3),
            b_factors=np.array([80.0, 90.0]),
            occupancies=np.ones(2),
        )

        with pytest.raises(ValueError, match="Either min_plddt or max_bfactor must be specified"):
            chain.filter_by_quality()
