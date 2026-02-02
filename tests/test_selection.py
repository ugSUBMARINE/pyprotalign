"""Tests for chain selection operations."""

import gemmi
import pytest

from pyprotalign.selection import compute_chain_center, get_all_protein_chains, get_chain


class TestGetChain:
    """Tests for get_chain function."""

    def test_none_returns_first_chain(self) -> None:
        """Test that None returns first protein chain."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add two chains
        chain_a = gemmi.Chain("A")
        res_a = gemmi.Residue()
        res_a.name = "ALA"
        res_a.seqid = gemmi.SeqId("1")
        res_a.entity_type = gemmi.EntityType.Polymer
        chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        chain_b = gemmi.Chain("B")
        res_b = gemmi.Residue()
        res_b.name = "GLY"
        res_b.seqid = gemmi.SeqId("1")
        res_b.entity_type = gemmi.EntityType.Polymer
        chain_b.add_residue(res_b)
        model.add_chain(chain_b)

        structure.add_model(model)
        structure.setup_entities()

        # Should return first chain (A)
        result = get_chain(structure, None)
        assert result.name == "A"

    def test_valid_chain_id(self) -> None:
        """Test retrieving chain by valid ID."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        res_a = gemmi.Residue()
        res_a.name = "ALA"
        res_a.seqid = gemmi.SeqId("1")
        res_a.entity_type = gemmi.EntityType.Polymer
        chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        chain_b = gemmi.Chain("B")
        res_b = gemmi.Residue()
        res_b.name = "GLY"
        res_b.seqid = gemmi.SeqId("1")
        res_b.entity_type = gemmi.EntityType.Polymer
        chain_b.add_residue(res_b)
        model.add_chain(chain_b)

        structure.add_model(model)
        structure.setup_entities()

        # Request chain B
        result = get_chain(structure, "B")
        assert result.name == "B"

    def test_invalid_chain_id(self) -> None:
        """Test error when chain ID not found."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        res_a = gemmi.Residue()
        res_a.name = "ALA"
        res_a.seqid = gemmi.SeqId("1")
        res_a.entity_type = gemmi.EntityType.Polymer
        chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        structure.add_model(model)
        structure.setup_entities()

        # Request non-existent chain
        with pytest.raises(ValueError, match="Chain 'Z' not found"):
            get_chain(structure, "Z")

    def test_non_protein_chain(self) -> None:
        """Test error when chain is not protein."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add chain with no polymer residues
        chain_a = gemmi.Chain("A")
        model.add_chain(chain_a)

        structure.add_model(model)
        structure.setup_entities()

        # Request chain with no polymer
        with pytest.raises(ValueError, match="Chain 'A' is not a protein chain"):
            get_chain(structure, "A")


class TestGetAllProteinChains:
    """Tests for get_all_protein_chains function."""

    def test_multiple_chains(self) -> None:
        """Test getting all protein chains from structure."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add three protein chains
        for chain_name in ["A", "B", "C"]:
            chain = gemmi.Chain(chain_name)
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId("1")
            res.entity_type = gemmi.EntityType.Polymer
            chain.add_residue(res)
            model.add_chain(chain)

        structure.add_model(model)
        structure.setup_entities()

        chains = get_all_protein_chains(structure)
        assert len(chains) == 3
        assert [c.name for c in chains] == ["A", "B", "C"]

    def test_single_chain(self) -> None:
        """Test getting single protein chain."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        res_a = gemmi.Residue()
        res_a.name = "ALA"
        res_a.seqid = gemmi.SeqId("1")
        res_a.entity_type = gemmi.EntityType.Polymer
        chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        structure.add_model(model)
        structure.setup_entities()

        chains = get_all_protein_chains(structure)
        assert len(chains) == 1
        assert chains[0].name == "A"

    def test_mixed_protein_nonprotein(self) -> None:
        """Test filtering out non-protein chains."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add protein chain
        chain_a = gemmi.Chain("A")
        res_a = gemmi.Residue()
        res_a.name = "ALA"
        res_a.seqid = gemmi.SeqId("1")
        res_a.entity_type = gemmi.EntityType.Polymer
        chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        # Add empty chain (non-protein)
        chain_b = gemmi.Chain("B")
        model.add_chain(chain_b)

        structure.add_model(model)
        structure.setup_entities()

        chains = get_all_protein_chains(structure)
        assert len(chains) == 1
        assert chains[0].name == "A"

    def test_no_protein_chains(self) -> None:
        """Test structure with no protein chains."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add only empty chains
        chain_a = gemmi.Chain("A")
        model.add_chain(chain_a)

        structure.add_model(model)
        structure.setup_entities()

        chains = get_all_protein_chains(structure)
        assert len(chains) == 0


class TestComputeChainCenter:
    """Tests for compute_chain_center function."""

    def test_normal_chain(self) -> None:
        """Test computing center for normal chain."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")

        # Add three residues with CA atoms at known positions
        for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0)]):
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(str(i + 1))
            res.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.pos = gemmi.Position(x, y, z)
            res.add_atom(atom)
            chain.add_residue(res)

        model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()

        center = compute_chain_center(chain)

        # Center should be (1.0, 1.0, 0.0)
        assert center.shape == (3,)
        assert center[0] == pytest.approx(1.0)
        assert center[1] == pytest.approx(1.0)
        assert center[2] == pytest.approx(0.0)

    def test_empty_chain(self) -> None:
        """Test error when chain has no CA atoms."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")

        model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()

        with pytest.raises(ValueError, match="Chain 'A' has no CA atoms"):
            compute_chain_center(chain)
