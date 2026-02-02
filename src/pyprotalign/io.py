"""Structure file I/O operations."""

from pathlib import Path

import gemmi


def load_structure(path: str) -> gemmi.Structure:
    """Load a protein structure from PDB or mmCIF file.

    Args:
        path: Path to structure file (PDB or mmCIF format)

    Returns:
        Loaded gemmi Structure object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If structure has no models
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    structure = gemmi.read_structure(str(file_path))

    if len(structure) == 0:
        raise ValueError(f"Structure has no models: {path}")

    # Setup entities to identify polymer chains
    structure.setup_entities()

    return structure


def write_structure(structure: gemmi.Structure, path: str) -> None:
    """Write structure to file with format auto-detection.

    Args:
        structure: gemmi Structure to write
        path: Output file path (.pdb or .cif extension)
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".cif":
        # Write mmCIF
        doc = structure.make_mmcif_document()
        doc.write_file(str(file_path))
    else:
        # Write PDB (default)
        structure.write_pdb(str(file_path))
