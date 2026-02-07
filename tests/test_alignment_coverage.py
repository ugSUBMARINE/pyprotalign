"""Additional tests for alignment.py coverage."""

import gemmi
import pytest

from pyprotalign.alignment import align_globally, align_quaternary, align_two_chains
from pyprotalign.gemmi_utils import get_all_protein_chains


def _create_chain(name: str, sequence: str, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> gemmi.Chain:
    """Helper to create a chain with sequence."""
    chain = gemmi.Chain(name)
    residues = sequence.split()
    for i, res_name in enumerate(residues):
        res = gemmi.Residue()
        res.name = res_name
        res.seqid = gemmi.SeqId(str(i + 1))
        res.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(float(i) + offset[0], offset[1], offset[2])
        atom.b_iso = 20.0  # B-factor
        atom.occ = 1.0  # Occupancy
        res.add_atom(atom)
        chain.add_residue(res)
    return chain


def _create_structure(chains: list[tuple[str, str, tuple[float, float, float]]]) -> gemmi.Structure:
    """Helper to create structure with multiple chains."""
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    for chain_name, sequence, offset in chains:
        chain = _create_chain(chain_name, sequence, offset)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    return structure


class TestAlignTwoChainsWithRefine:
    """Test align_two_chains with refine=True."""

    def test_with_refinement(self) -> None:
        """Test alignment with iterative refinement enabled."""
        fixed_st = _create_structure([("A", "ALA GLY SER THR VAL", (0.0, 0.0, 0.0))])
        mobile_st = _create_structure([("A", "ALA GLY SER THR VAL", (0.0, 0.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned = align_two_chains(
            fixed_chains[0], mobile_chains[0], refine=True, max_cycles=5, cutoff_factor=2.0
        )

        assert num_aligned == 5
        assert rmsd < 0.01  # Identical structures


class TestAlignTwoChainsWithFilter:
    """Test align_two_chains with quality filtering."""

    def test_with_bfactor_filter(self) -> None:
        """Test alignment with B-factor filtering."""
        fixed_st = _create_structure([("A", "ALA GLY SER THR VAL", (0.0, 0.0, 0.0))])
        mobile_st = _create_structure([("A", "ALA GLY SER THR VAL", (0.0, 0.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned = align_two_chains(
            fixed_chains[0],
            mobile_chains[0],
            filter=True,
            min_bfactor=0.0,
            max_bfactor=50.0,
            min_occ=0.5,
        )

        assert num_aligned == 5

    def test_filter_insufficient_pairs_error(self) -> None:
        """Test error when filtering leaves < 3 pairs."""
        fixed_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0))])
        mobile_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        # Filter with impossible threshold to reject all atoms
        with pytest.raises(ValueError, match="Need at least 3 CA pairs after quality filtering"):
            align_two_chains(
                fixed_chains[0],
                mobile_chains[0],
                filter=True,
                min_bfactor=1000.0,  # Impossibly high threshold
                max_bfactor=2000.0,
                min_occ=0.0,
            )


class TestAlignGloballyByOrder:
    """Test align_globally with by_chain_id=False."""

    def test_align_by_order(self) -> None:
        """Test aligning chains by their order instead of ID."""
        # Fixed: A, B, C; Mobile: X, Y, Z (different IDs, same order)
        fixed_st = _create_structure(
            [
                ("A", "ALA GLY SER", (0.0, 0.0, 0.0)),
                ("B", "THR VAL", (0.0, 10.0, 0.0)),
                ("C", "LEU ILE", (0.0, 20.0, 0.0)),
            ]
        )
        mobile_st = _create_structure(
            [
                ("X", "ALA GLY SER", (0.0, 0.0, 0.0)),
                ("Y", "THR VAL", (0.0, 10.0, 0.0)),
                ("Z", "LEU ILE", (0.0, 20.0, 0.0)),
            ]
        )

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(
            fixed_map, mobile_map, by_chain_id=False
        )

        # Should match by order: A->X, B->Y, C->Z
        assert chain_mapping == {"A": "X", "B": "Y", "C": "Z"}
        assert num_aligned == 7  # 3+2+2 residues


class TestAlignQuaternaryErrors:
    """Test error conditions in align_quaternary."""

    def test_invalid_fixed_seed_chain(self) -> None:
        """Test error when fixed seed chain doesn't exist."""
        fixed_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0))])
        mobile_st = _create_structure([("B", "ALA GLY SER", (0.0, 0.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        with pytest.raises(ValueError, match="Fixed seed chain 'Z' not found"):
            align_quaternary(fixed_chains, mobile_chains, fixed_seed="Z")

    def test_invalid_mobile_seed_chain(self) -> None:
        """Test error when mobile seed chain doesn't exist."""
        fixed_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0))])
        mobile_st = _create_structure([("B", "ALA GLY SER", (0.0, 0.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        with pytest.raises(ValueError, match="Mobile seed chain 'Z' not found"):
            align_quaternary(fixed_chains, mobile_chains, mobile_seed="Z")


class TestAlignWithGaps:
    """Test alignment with gaps in sequence alignment."""

    def test_sequence_with_gaps(self) -> None:
        """Test alignment when sequences have gaps."""
        # Create chains with different sequences to force gaps
        fixed_st = _create_structure([("A", "ALA GLY SER THR VAL LEU", (0.0, 0.0, 0.0))])
        mobile_st = _create_structure([("A", "ALA GLY THR VAL", (0.0, 0.0, 0.0))])  # Missing SER, LEU

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned = align_two_chains(fixed_chains[0], mobile_chains[0])

        # Should align 4 residues (gaps for SER and LEU)
        assert num_aligned == 4


class TestAlignMappedChainsWithWarnings:
    """Test warning paths in align_mapped_chains."""

    def test_chain_with_no_aligned_pairs(self) -> None:
        """Test warning when a chain pair has no aligned CA atoms."""
        # Create structures where one chain has very different sequence
        fixed_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0)), ("B", "THR VAL LEU", (0.0, 10.0, 0.0))])
        mobile_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0)), ("B", "PHE TRP TYR", (0.0, 10.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        # Should work, but might have warning for chain B if sequences don't align well
        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(fixed_map, mobile_map)

        # Should still succeed with chain A
        assert num_aligned >= 3


class TestRefineWithFilter:
    """Test combination of refine + filter."""

    def test_refine_and_filter_together(self) -> None:
        """Test alignment with both refinement and filtering."""
        fixed_st = _create_structure([("A", "ALA GLY SER THR VAL LEU", (0.0, 0.0, 0.0))])
        mobile_st = _create_structure([("A", "ALA GLY SER THR VAL LEU", (0.0, 0.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])

        rotation, translation, rmsd, num_aligned = align_two_chains(
            fixed_chains[0],
            mobile_chains[0],
            refine=True,
            max_cycles=3,
            cutoff_factor=2.0,
            filter=True,
            min_bfactor=0.0,
            max_bfactor=50.0,
            min_occ=0.5,
        )

        assert num_aligned == 6
        assert rmsd < 0.01


class TestAlignGloballyWithRefine:
    """Test align_globally with refinement."""

    def test_global_with_refine(self) -> None:
        """Test global alignment with refinement enabled."""
        fixed_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0)), ("B", "THR VAL", (0.0, 10.0, 0.0))])
        mobile_st = _create_structure([("A", "ALA GLY SER", (0.0, 0.0, 0.0)), ("B", "THR VAL", (0.0, 10.0, 0.0))])

        fixed_chains = get_all_protein_chains(fixed_st[0])
        mobile_chains = get_all_protein_chains(mobile_st[0])
        fixed_map = {c.chain_id: c for c in fixed_chains}
        mobile_map = {c.chain_id: c for c in mobile_chains}

        rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(
            fixed_map, mobile_map, refine=True, max_cycles=5, cutoff_factor=2.0
        )

        assert num_aligned == 5
        assert rmsd < 0.01
