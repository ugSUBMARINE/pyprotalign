"""Tests for structure I/O operations."""

from pathlib import Path

import gemmi
import pytest

from pyprotalign.io import load_structure, write_structure


class TestLoadStructure:
    """Tests for load_structure function."""

    def test_load_pdb_file(self, tmp_path: Path) -> None:
        """Test loading a PDB file."""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )

        structure = load_structure(str(pdb_path))
        assert len(structure) == 1
        assert len(structure[0]) > 0

    def test_load_cif_file(self, tmp_path: Path) -> None:
        """Test loading an mmCIF file."""
        cif_path = tmp_path / "test.cif"
        cif_content = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 ? 0.000 0.000 0.000 1.00 0.00 ? 1 ALA A CA 1
ATOM 2 C CA . GLY A 1 2 ? 3.800 0.000 0.000 1.00 0.00 ? 2 GLY A CA 1
"""
        cif_path.write_text(cif_content)

        structure = load_structure(str(cif_path))
        assert len(structure) == 1
        assert len(structure[0]) > 0

    def test_file_not_found_error(self) -> None:
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError, match="Structure file not found"):
            load_structure("/nonexistent/path/file.pdb")


class TestWriteStructure:
    """Tests for write_structure function."""

    def test_write_pdb(self, tmp_path: Path) -> None:
        """Test writing PDB file."""
        structure = gemmi.Structure()
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId("1")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(0.0, 0.0, 0.0)
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)

        output_path = tmp_path / "output.pdb"
        write_structure(structure, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "CA" in content
        assert "ALA" in content

    def test_write_cif(self, tmp_path: Path) -> None:
        """Test writing mmCIF file."""
        structure = gemmi.Structure()
        structure.name = "test"
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId("1")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(0.0, 0.0, 0.0)
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()

        output_path = tmp_path / "output.cif"
        write_structure(structure, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "data_" in content
        assert "_atom_site" in content

    def test_format_detection_by_extension(self, tmp_path: Path) -> None:
        """Test automatic format detection from file extension."""
        structure = gemmi.Structure()
        structure.name = "test"
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId("1")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(0.0, 0.0, 0.0)
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()

        # PDB format
        pdb_path = tmp_path / "test.pdb"
        write_structure(structure, str(pdb_path))
        pdb_content = pdb_path.read_text()
        assert "CA" in pdb_content

        # CIF format
        cif_path = tmp_path / "test.cif"
        write_structure(structure, str(cif_path))
        cif_content = cif_path.read_text()
        assert "data_" in cif_content
