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


class TestFilterCaAtomsByQuality:
    """Tests for filter_ca_atoms_by_quality function."""

    def test_plddt_filtering_keeps_high_values(self) -> None:
        """Test that pLDDT filter keeps atoms with high pLDDT."""
        # Create atoms with different pLDDT values (stored in b_iso)
        atoms: list[gemmi.Atom | None] = []
        for plddt in [90.0, 70.0, 50.0, 30.0]:
            atom = gemmi.Atom()
            atom.b_iso = plddt
            atoms.append(atom)

        from pyprotalign.selection import filter_ca_atoms_by_quality

        mask = filter_ca_atoms_by_quality(atoms, min_plddt=60.0)

        # Should keep atoms with pLDDT >= 60.0
        assert mask[0]  # 90.0
        assert mask[1]  # 70.0
        assert not mask[2]  # 50.0
        assert not mask[3]  # 30.0

    def test_bfactor_filtering_keeps_low_values(self) -> None:
        """Test that B-factor filter keeps atoms with low B-factor."""
        atoms: list[gemmi.Atom | None] = []
        for bfactor in [10.0, 30.0, 50.0, 70.0]:
            atom = gemmi.Atom()
            atom.b_iso = bfactor
            atoms.append(atom)

        from pyprotalign.selection import filter_ca_atoms_by_quality

        mask = filter_ca_atoms_by_quality(atoms, max_bfactor=40.0)

        # Should keep atoms with B-factor <= 40.0
        assert mask[0]  # 10.0
        assert mask[1]  # 30.0
        assert not mask[2]  # 50.0
        assert not mask[3]  # 70.0

    def test_mutual_exclusivity_error(self) -> None:
        """Test error when both filters specified."""
        atoms = [gemmi.Atom()]

        from pyprotalign.selection import filter_ca_atoms_by_quality

        with pytest.raises(ValueError, match="mutually exclusive"):
            filter_ca_atoms_by_quality(atoms, min_plddt=50.0, max_bfactor=30.0)

    def test_neither_specified_error(self) -> None:
        """Test error when neither filter specified."""
        atoms = [gemmi.Atom()]

        from pyprotalign.selection import filter_ca_atoms_by_quality

        with pytest.raises(ValueError, match="Either min_plddt or max_bfactor must be specified"):
            filter_ca_atoms_by_quality(atoms)

    def test_none_atoms_filtered_out(self) -> None:
        """Test that None atoms are always filtered out."""
        atoms: list[gemmi.Atom | None] = []
        atom1 = gemmi.Atom()
        atom1.b_iso = 90.0
        atoms.append(atom1)
        atoms.append(None)
        atom3 = gemmi.Atom()
        atom3.b_iso = 80.0
        atoms.append(atom3)

        from pyprotalign.selection import filter_ca_atoms_by_quality

        mask = filter_ca_atoms_by_quality(atoms, min_plddt=60.0)

        assert mask[0]
        assert not mask[1]  # None filtered out
        assert mask[2]

    def test_all_none_atoms(self) -> None:
        """Test filtering with all None atoms."""
        atoms: list[gemmi.Atom | None] = [None, None, None]

        from pyprotalign.selection import filter_ca_atoms_by_quality

        mask = filter_ca_atoms_by_quality(atoms, min_plddt=50.0)

        assert not any(mask)

    def test_edge_values_plddt(self) -> None:
        """Test edge values for pLDDT filtering."""
        atoms: list[gemmi.Atom | None] = []
        for plddt in [0.0, 50.0, 100.0]:
            atom = gemmi.Atom()
            atom.b_iso = plddt
            atoms.append(atom)

        from pyprotalign.selection import filter_ca_atoms_by_quality

        # Test threshold at 50.0
        mask = filter_ca_atoms_by_quality(atoms, min_plddt=50.0)
        assert not mask[0]  # 0.0 < 50.0
        assert mask[1]  # 50.0 >= 50.0 (equality edge case)
        assert mask[2]  # 100.0 >= 50.0

    def test_edge_values_bfactor(self) -> None:
        """Test edge values for B-factor filtering."""
        atoms: list[gemmi.Atom | None] = []
        for bfactor in [0.0, 50.0, 100.0]:
            atom = gemmi.Atom()
            atom.b_iso = bfactor
            atoms.append(atom)

        from pyprotalign.selection import filter_ca_atoms_by_quality

        # Test threshold at 50.0
        mask = filter_ca_atoms_by_quality(atoms, max_bfactor=50.0)
        assert mask[0]  # 0.0 <= 50.0
        assert mask[1]  # 50.0 <= 50.0 (equality edge case)
        assert not mask[2]  # 100.0 > 50.0


class TestFilterCaPairsByQuality:
    """Tests for filter_ca_pairs_by_quality function."""

    def test_both_atoms_must_pass(self) -> None:
        """Test that both atoms in pair must pass filter."""
        # Fixed: all high pLDDT
        fixed_atoms: list[gemmi.Atom | None] = []
        for plddt in [90.0, 80.0, 70.0]:
            atom = gemmi.Atom()
            atom.b_iso = plddt
            fixed_atoms.append(atom)

        # Mobile: mixed quality
        mobile_atoms: list[gemmi.Atom | None] = []
        for plddt in [85.0, 40.0, 75.0]:  # Second one fails threshold
            atom = gemmi.Atom()
            atom.b_iso = plddt
            mobile_atoms.append(atom)

        from pyprotalign.selection import filter_ca_pairs_by_quality

        mask = filter_ca_pairs_by_quality(fixed_atoms, mobile_atoms, min_plddt=60.0)

        # First pair: both pass (90, 85)
        assert mask[0]
        # Second pair: mobile fails (80, 40)
        assert not mask[1]
        # Third pair: both pass (70, 75)
        assert mask[2]

    def test_one_atom_fails_pair_rejected(self) -> None:
        """Test that if one atom fails, entire pair is rejected."""
        fixed_atoms: list[gemmi.Atom | None] = []
        atom1 = gemmi.Atom()
        atom1.b_iso = 40.0  # Fails
        fixed_atoms.append(atom1)

        mobile_atoms: list[gemmi.Atom | None] = []
        atom2 = gemmi.Atom()
        atom2.b_iso = 80.0  # Passes
        mobile_atoms.append(atom2)

        from pyprotalign.selection import filter_ca_pairs_by_quality

        mask = filter_ca_pairs_by_quality(fixed_atoms, mobile_atoms, min_plddt=60.0)

        # Fixed fails, mobile passes → pair rejected
        assert not mask[0]

    def test_index_relationship_preserved(self) -> None:
        """Test that index relationship is preserved."""
        fixed_atoms: list[gemmi.Atom | None] = []
        mobile_atoms: list[gemmi.Atom | None] = []

        # Create 5 pairs with various quality combinations
        fixed_plddt = [90.0, 50.0, 80.0, 30.0, 70.0]
        mobile_plddt = [85.0, 75.0, 40.0, 65.0, 68.0]

        for plddt in fixed_plddt:
            atom = gemmi.Atom()
            atom.b_iso = plddt
            fixed_atoms.append(atom)

        for plddt in mobile_plddt:
            atom = gemmi.Atom()
            atom.b_iso = plddt
            mobile_atoms.append(atom)

        from pyprotalign.selection import filter_ca_pairs_by_quality

        mask = filter_ca_pairs_by_quality(fixed_atoms, mobile_atoms, min_plddt=60.0)

        # Pair 0: (90, 85) → both pass
        assert mask[0]
        # Pair 1: (50, 75) → fixed fails
        assert not mask[1]
        # Pair 2: (80, 40) → mobile fails
        assert not mask[2]
        # Pair 3: (30, 65) → fixed fails
        assert not mask[3]
        # Pair 4: (70, 68) → both pass
        assert mask[4]

        # Mask length equals input length
        assert len(mask) == len(fixed_atoms) == len(mobile_atoms)

    def test_length_mismatch_error(self) -> None:
        """Test error when lists have different lengths."""
        fixed_atoms = [gemmi.Atom(), gemmi.Atom()]
        mobile_atoms = [gemmi.Atom()]

        from pyprotalign.selection import filter_ca_pairs_by_quality

        with pytest.raises(ValueError, match="must have same length"):
            filter_ca_pairs_by_quality(fixed_atoms, mobile_atoms, min_plddt=50.0)

    def test_none_atoms_in_pairs(self) -> None:
        """Test handling of None atoms in pairs."""
        fixed_atoms: list[gemmi.Atom | None] = []
        mobile_atoms: list[gemmi.Atom | None] = []

        # Pair 0: both valid, high quality
        atom1 = gemmi.Atom()
        atom1.b_iso = 90.0
        fixed_atoms.append(atom1)
        atom2 = gemmi.Atom()
        atom2.b_iso = 85.0
        mobile_atoms.append(atom2)

        # Pair 1: fixed is None
        fixed_atoms.append(None)
        atom3 = gemmi.Atom()
        atom3.b_iso = 80.0
        mobile_atoms.append(atom3)

        # Pair 2: mobile is None
        atom4 = gemmi.Atom()
        atom4.b_iso = 75.0
        fixed_atoms.append(atom4)
        mobile_atoms.append(None)

        from pyprotalign.selection import filter_ca_pairs_by_quality

        mask = filter_ca_pairs_by_quality(fixed_atoms, mobile_atoms, min_plddt=60.0)

        # Pair 0: both valid and pass
        assert mask[0]
        # Pair 1: fixed is None → fails
        assert not mask[1]
        # Pair 2: mobile is None → fails
        assert not mask[2]
