"""Tests for chain selection operations."""

import gemmi
import pytest

from pyprotalign.gemmi_utils import get_all_protein_chains, get_chain


class TestGetChain:
    """Tests for get_chain function."""

    def test_none_returns_first_chain(self) -> None:
        """Test that None returns first protein chain."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add two chains with multiple residues + CA atoms
        chain_a = gemmi.Chain("A")
        for i, res_name in enumerate(["ALA", "GLY", "SER"]):
            res_a = gemmi.Residue()
            res_a.name = res_name
            res_a.seqid = gemmi.SeqId(str(i + 1))
            res_a.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res_a.add_atom(atom)
            chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        chain_b = gemmi.Chain("B")
        for i, res_name in enumerate(["VAL", "LEU"]):
            res_b = gemmi.Residue()
            res_b.name = res_name
            res_b.seqid = gemmi.SeqId(str(i + 1))
            res_b.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res_b.add_atom(atom)
            chain_b.add_residue(res_b)
        model.add_chain(chain_b)

        structure.add_model(model)
        structure.setup_entities()

        # Should return first chain (A)
        result = get_chain(structure[0], None)
        assert result.chain_id == "A"

    def test_valid_chain_id(self) -> None:
        """Test retrieving chain by valid ID."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        for i, res_name in enumerate(["ALA", "GLY"]):
            res_a = gemmi.Residue()
            res_a.name = res_name
            res_a.seqid = gemmi.SeqId(str(i + 1))
            res_a.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res_a.add_atom(atom)
            chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        chain_b = gemmi.Chain("B")
        for i, res_name in enumerate(["SER", "THR"]):
            res_b = gemmi.Residue()
            res_b.name = res_name
            res_b.seqid = gemmi.SeqId(str(i + 1))
            res_b.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res_b.add_atom(atom)
            chain_b.add_residue(res_b)
        model.add_chain(chain_b)

        structure.add_model(model)
        structure.setup_entities()

        # Request chain B
        result = get_chain(structure[0], "B")
        assert result.chain_id == "B"

    def test_invalid_chain_id(self) -> None:
        """Test error when chain ID not found."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        for i, res_name in enumerate(["ALA", "GLY"]):
            res_a = gemmi.Residue()
            res_a.name = res_name
            res_a.seqid = gemmi.SeqId(str(i + 1))
            res_a.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res_a.add_atom(atom)
            chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        structure.add_model(model)
        structure.setup_entities()

        # Request non-existent chain
        with pytest.raises(ValueError, match="No protein chain 'Z' found in structure"):
            get_chain(structure[0], "Z")

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
        with pytest.raises(ValueError, match="No protein chain 'A' found in structure"):
            get_chain(structure[0], "A")


class TestGetAllProteinChains:
    """Tests for get_all_protein_chains function."""

    def test_multiple_chains(self) -> None:
        """Test getting all protein chains from structure."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add three protein chains with multiple residues + CA atoms
        for chain_name in ["A", "B", "C"]:
            chain = gemmi.Chain(chain_name)
            for i, res_name in enumerate(["ALA", "GLY"]):
                res = gemmi.Residue()
                res.name = res_name
                res.seqid = gemmi.SeqId(str(i + 1))
                res.entity_type = gemmi.EntityType.Polymer
                atom = gemmi.Atom()
                atom.name = "CA"
                atom.element = gemmi.Element("C")
                atom.pos = gemmi.Position(float(i), 0.0, 0.0)
                res.add_atom(atom)
                chain.add_residue(res)
            model.add_chain(chain)

        structure.add_model(model)
        structure.setup_entities()

        chains = get_all_protein_chains(structure[0])
        assert len(chains) == 3
        assert [c.chain_id for c in chains] == ["A", "B", "C"]

    def test_single_chain(self) -> None:
        """Test getting single protein chain."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        for i, res_name in enumerate(["ALA", "GLY"]):
            res_a = gemmi.Residue()
            res_a.name = res_name
            res_a.seqid = gemmi.SeqId(str(i + 1))
            res_a.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res_a.add_atom(atom)
            chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        structure.add_model(model)
        structure.setup_entities()

        chains = get_all_protein_chains(structure[0])
        assert len(chains) == 1
        assert chains[0].chain_id == "A"

    def test_mixed_protein_nonprotein(self) -> None:
        """Test filtering out non-protein chains."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add protein chain with multiple residues + CA atoms
        chain_a = gemmi.Chain("A")
        for i, res_name in enumerate(["ALA", "GLY"]):
            res_a = gemmi.Residue()
            res_a.name = res_name
            res_a.seqid = gemmi.SeqId(str(i + 1))
            res_a.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res_a.add_atom(atom)
            chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        # Add empty chain (non-protein)
        chain_b = gemmi.Chain("B")
        model.add_chain(chain_b)

        structure.add_model(model)
        structure.setup_entities()

        chains = get_all_protein_chains(structure[0])
        assert len(chains) == 1
        assert chains[0].chain_id == "A"

    def test_no_protein_chains(self) -> None:
        """Test structure with no protein chains."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        # Add only empty chains
        chain_a = gemmi.Chain("A")
        model.add_chain(chain_a)

        structure.add_model(model)
        structure.setup_entities()

        with pytest.raises(ValueError, match="No protein chains found in model"):
            get_all_protein_chains(structure[0])
