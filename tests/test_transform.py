"""Tests for coordinate transformation."""

import gemmi
import numpy as np
import pytest

from pyprotalign.transform import apply_transformation, generate_conflict_free_chain_map, rename_chains


class TestApplyTransformation:
    """Tests for apply_transformation function."""

    def test_identity_transformation(self) -> None:
        """Test identity transformation leaves coordinates unchanged."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")

        # Add residue with atom
        res = gemmi.Residue()
        res.name = "ALA"
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(1.0, 2.0, 3.0)
        res.add_atom(atom)
        chain.add_residue(res)
        model.add_chain(chain)
        structure.add_model(model)

        # Identity transformation
        rotation = np.eye(3)
        translation = np.zeros(3)

        apply_transformation(structure, rotation, translation)

        # Check position unchanged
        ca = structure[0][0][0].find_atom("CA", "*")
        assert ca is not None
        assert ca.pos.x == 1.0
        assert ca.pos.y == 2.0
        assert ca.pos.z == 3.0

    def test_translation(self) -> None:
        """Test pure translation."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")

        res = gemmi.Residue()
        res.name = "ALA"
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(1.0, 2.0, 3.0)
        res.add_atom(atom)
        chain.add_residue(res)
        model.add_chain(chain)
        structure.add_model(model)

        rotation = np.eye(3)
        translation = np.array([10.0, 20.0, 30.0])

        apply_transformation(structure, rotation, translation)

        ca = structure[0][0][0].find_atom("CA", "*")
        assert ca is not None
        assert ca.pos.x == 11.0
        assert ca.pos.y == 22.0
        assert ca.pos.z == 33.0

    def test_rotation_90_degrees(self) -> None:
        """Test 90-degree rotation around Z-axis."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")

        res = gemmi.Residue()
        res.name = "ALA"
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(1.0, 0.0, 0.0)
        res.add_atom(atom)
        chain.add_residue(res)
        model.add_chain(chain)
        structure.add_model(model)

        # 90-degree rotation around Z: (x,y,z) -> (-y,x,z)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.zeros(3)

        apply_transformation(structure, rotation, translation)

        ca = structure[0][0][0].find_atom("CA", "*")
        assert ca is not None
        assert ca.pos.x == pytest.approx(0.0, abs=1e-10)
        assert ca.pos.y == pytest.approx(1.0, abs=1e-10)
        assert ca.pos.z == pytest.approx(0.0, abs=1e-10)

    def test_multiple_atoms(self) -> None:
        """Test transformation applies to all atoms."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")

        # Add residue with multiple atoms
        res = gemmi.Residue()
        res.name = "ALA"

        for i, atom_name in enumerate(["N", "CA", "C", "O"]):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            res.add_atom(atom)

        chain.add_residue(res)
        model.add_chain(chain)
        structure.add_model(model)

        rotation = np.eye(3)
        translation = np.array([5.0, 0.0, 0.0])

        apply_transformation(structure, rotation, translation)

        # Check all atoms transformed
        for i, atom_name in enumerate(["N", "CA", "C", "O"]):
            atom = structure[0][0][0].find_atom(atom_name, "*")
            assert atom is not None
            assert atom.pos.x == pytest.approx(float(i) + 5.0, abs=1e-10)

    def test_multiple_chains(self) -> None:
        """Test transformation applies to all chains."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        for chain_name in ["A", "B"]:
            chain = gemmi.Chain(chain_name)
            res = gemmi.Residue()
            res.name = "ALA"
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(1.0, 0.0, 0.0)
            res.add_atom(atom)
            chain.add_residue(res)
            model.add_chain(chain)

        structure.add_model(model)

        rotation = np.eye(3)
        translation = np.array([10.0, 0.0, 0.0])

        apply_transformation(structure, rotation, translation)

        # Check both chains transformed
        for chain in structure[0]:
            ca = chain[0].find_atom("CA", "*")
            assert ca is not None
            assert ca.pos.x == pytest.approx(11.0, abs=1e-10)


class TestRenameChains:
    """Tests for rename_chains function."""

    def test_simple_rename(self) -> None:
        """Test simple chain renaming."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        res_a = gemmi.Residue()
        res_a.name = "ALA"
        atom_a = gemmi.Atom()
        atom_a.name = "CA"
        atom_a.pos = gemmi.Position(1.0, 2.0, 3.0)
        res_a.add_atom(atom_a)
        chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        chain_b = gemmi.Chain("B")
        res_b = gemmi.Residue()
        res_b.name = "GLY"
        atom_b = gemmi.Atom()
        atom_b.name = "CA"
        atom_b.pos = gemmi.Position(4.0, 5.0, 6.0)
        res_b.add_atom(atom_b)
        chain_b.add_residue(res_b)
        model.add_chain(chain_b)

        structure.add_model(model)

        # Rename A->X, B->Y
        rename_chains(structure, {"A": "X", "B": "Y"})

        chain_names = [chain.name for chain in structure[0]]
        assert chain_names == ["X", "Y"]

    def test_chain_name_collision(self) -> None:
        """Test renaming with name collision (A->B, B->A swap)."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        chain_a = gemmi.Chain("A")
        res_a = gemmi.Residue()
        res_a.name = "ALA"
        atom_a = gemmi.Atom()
        atom_a.name = "CA"
        atom_a.pos = gemmi.Position(1.0, 2.0, 3.0)
        res_a.add_atom(atom_a)
        chain_a.add_residue(res_a)
        model.add_chain(chain_a)

        chain_b = gemmi.Chain("B")
        res_b = gemmi.Residue()
        res_b.name = "GLY"
        atom_b = gemmi.Atom()
        atom_b.name = "CA"
        atom_b.pos = gemmi.Position(4.0, 5.0, 6.0)
        res_b.add_atom(atom_b)
        chain_b.add_residue(res_b)
        model.add_chain(chain_b)

        structure.add_model(model)

        # Swap A and B
        rename_chains(structure, {"A": "B", "B": "A"})

        chain_names = [chain.name for chain in structure[0]]
        assert chain_names == ["B", "A"]

        # Check residue types preserved
        assert structure[0][0][0].name == "ALA"  # First chain (renamed to B)
        assert structure[0][1][0].name == "GLY"  # Second chain (renamed to A)

    def test_empty_map(self) -> None:
        """Test that empty map does nothing."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain_a = gemmi.Chain("A")
        model.add_chain(chain_a)
        structure.add_model(model)

        rename_chains(structure, {})

        assert structure[0][0].name == "A"

    def test_partial_rename(self) -> None:
        """Test renaming only some chains."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        for chain_name in ["A", "B", "C"]:
            chain = gemmi.Chain(chain_name)
            res = gemmi.Residue()
            res.name = "ALA"
            chain.add_residue(res)
            model.add_chain(chain)

        structure.add_model(model)

        # Only rename A->X
        rename_chains(structure, {"A": "X"})

        chain_names = [chain.name for chain in structure[0]]
        assert chain_names == ["X", "B", "C"]


class TestGenerateConflictFreeChainMap:
    """Tests for generate_conflict_free_chain_map function."""

    def test_no_conflicts(self) -> None:
        """Test case with no naming conflicts."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        for chain_name in ["A", "B", "C"]:
            chain = gemmi.Chain(chain_name)
            model.add_chain(chain)

        structure.add_model(model)

        # Align A->X, B->Y (no conflict with existing C)
        chain_pairs = [("X", "A"), ("Y", "B")]
        chain_map = generate_conflict_free_chain_map(structure, chain_pairs)

        assert chain_map == {"A": "X", "B": "Y"}

    def test_simple_conflict(self) -> None:
        """Test conflict resolution with unaligned chain."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        for chain_name in ["A", "B", "C", "D"]:
            chain = gemmi.Chain(chain_name)
            model.add_chain(chain)

        structure.add_model(model)

        # Align A->B, D->D (conflict: B exists as unaligned chain)
        chain_pairs = [("B", "A"), ("D", "D")]
        chain_map = generate_conflict_free_chain_map(structure, chain_pairs)

        # Should add swap: B->A, skip identity D->D
        assert chain_map == {"A": "B", "B": "A"}

    def test_multiple_conflicts(self) -> None:
        """Test multiple conflict resolution."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        for chain_name in ["A", "B", "C", "D"]:
            chain = gemmi.Chain(chain_name)
            model.add_chain(chain)

        structure.add_model(model)

        # Align C->A, D->B (conflicts: both A and B exist as unaligned)
        chain_pairs = [("A", "C"), ("B", "D")]
        chain_map = generate_conflict_free_chain_map(structure, chain_pairs)

        # Should add swaps: A->C, B->D
        assert chain_map == {"C": "A", "A": "C", "D": "B", "B": "D"}

    def test_no_conflict_when_target_is_aligned(self) -> None:
        """Test no swap when target chain is also being aligned."""
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        for chain_name in ["A", "B"]:
            chain = gemmi.Chain(chain_name)
            model.add_chain(chain)

        structure.add_model(model)

        # Align A->B, B->C where C doesn't exist in mobile
        # This tests that B being aligned means no swap even though B exists
        chain_pairs = [("B", "A"), ("C", "B")]
        chain_map = generate_conflict_free_chain_map(structure, chain_pairs)

        # Should not add swap for B since it's being renamed to C
        assert chain_map == {"A": "B", "B": "C"}

    def test_real_world_9jn4_9jn5(self) -> None:
        """Test with real structures 9jn4/9jn5 case."""
        # Simulate 9jn5 mobile structure: chains A, B, C, D
        structure = gemmi.Structure()
        model = gemmi.Model(1)

        for chain_name in ["A", "B", "C", "D"]:
            chain = gemmi.Chain(chain_name)
            model.add_chain(chain)

        structure.add_model(model)

        # Real alignment: B(fixed)->A(mobile), D(fixed)->D(mobile)
        chain_pairs = [("B", "A"), ("D", "D")]
        chain_map = generate_conflict_free_chain_map(structure, chain_pairs)

        # Should swap B->A to avoid conflict, skip identity D->D
        assert chain_map == {"A": "B", "B": "A"}

        # Apply renaming and verify no duplicates
        rename_chains(structure, chain_map)
        chain_names = [chain.name for chain in structure[0]]
        unique_names = set(chain_names)

        assert len(chain_names) == len(unique_names)  # No duplicates
        assert unique_names == {"A", "B", "C", "D"}  # All chains present
