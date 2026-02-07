"""Utility functions for working with Biopython library in pyprotalign package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

try:
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.mmcifio import MMCIFIO
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.Model import Model
    from Bio.PDB.PDBIO import PDBIO
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder, is_aa
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure
    from Bio.SeqUtils import seq1

except ImportError as e:
    raise ImportError("Biopython is required. Install with: pip install biopython") from e

import numpy as np
from numpy.typing import NDArray

from .chain import ProteinChain

logger = logging.getLogger(__name__)


def create_chain(chain: Chain, chain_id: str | None = None) -> ProteinChain:
    """Create ProteinChain from Biopython Chain object.

    Args:
        chain: Biopython Chain object (Bio.PDB.Chain.Chain)
        chain_id: Optional chain ID override (uses chain.id if None)

    Returns:
        ProteinChain instance

    Raises:
        ValueError: If chain has no residues
        ImportError: If Biopython is not installed

    Note:
        Alternate locations (altlocs) are handled by Biopython's default
        selection behavior. In practice, this tends to pick the atom with
        higher occupancy, even if it is not altloc "A". This can differ from
        the Gemmi implementation in gemmi_utils.create_chain.
    """
    # Get chain ID
    if chain_id is None:
        chain_id = chain.id

    # Build polypeptides (handles missing residues)
    ppb = PPBuilder()  # type: ignore[no-untyped-call]
    polypeptides = ppb.build_peptides(chain)  # type: ignore[no-untyped-call]

    if len(polypeptides) == 0:
        raise ValueError(f"Chain '{chain_id}' has no polypeptide segments")

    # Collect residues from all polypeptide segments
    residues = []
    for pp in polypeptides:
        residues.extend(pp)

    if len(residues) == 0:
        raise ValueError(f"Chain '{chain_id}' has no amino acid residues")

    # Extract sequence and data
    sequence_chars = []
    coords_list = []
    b_factors_list = []
    occupancies_list = []

    for residue in residues:
        residue = cast(Residue, residue)
        # Get single-letter code
        if is_aa(residue):  # type: ignore[no-untyped-call]
            sequence_chars.append(residue.get_resname())  # type: ignore[no-untyped-call]
        else:
            # Skip non-standard residues
            continue

        # Extract CA atom if present
        if "CA" in residue:
            ca_atom = residue["CA"]
            coords_list.append(ca_atom.coord)
            b_factors_list.append(ca_atom.bfactor)
            occupancies_list.append(ca_atom.occupancy)
        else:
            # Missing CA atom
            coords_list.append([np.nan, np.nan, np.nan])
            b_factors_list.append(np.nan)
            occupancies_list.append(np.nan)

    # Convert 3-letter codes to 1-letter
    sequence = seq1("".join(sequence_chars))  # type: ignore[no-untyped-call]

    coords = np.array(coords_list, dtype=float)
    b_factors = np.array(b_factors_list, dtype=float)
    occupancies = np.array(occupancies_list, dtype=float)

    return ProteinChain(
        chain_id=chain_id,
        sequence=sequence,
        coords=coords,
        b_factors=b_factors,
        occupancies=occupancies,
    )


def align_sequences(seq_1: str, seq_2: str) -> list[tuple[int | None, int | None]]:
    """Align two sequences and return paired indices.

    Args:
        seq_1: First sequence (fixed)
        seq_2: Second sequence (mobile)

    Returns:
        List of paired indices. Each tuple contains (seq1_idx, seq2_idx).
        Gaps are indicated by 'None'.

    Note:
        Uses Biopython's PairwiseAligner in 'global' mode with BLOSUM62 scoring,
        'open_gap_score' -10, and 'extend_gap_score' -0.5.
    """
    aligner = PairwiseAligner(  # type: ignore[no-untyped-call]
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),  # type: ignore[no-untyped-call]
        open_gap_score=-10,
        extend_gap_score=-0.5,
    )
    alignment = aligner.align(seq_1, seq_2)[0]  # type: ignore[no-untyped-call]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("\nSequence alignment:\n")
        for line in str(alignment).strip().split("\n"):
            logger.debug(line.replace("target", "fixed ").replace("query ", "mobile"))

        logger.debug("\nScore: %.0f", alignment.score)
        logger.debug("Indentity: %.1f%%\n", alignment.counts().identities / len(seq_1) * 100.0)

    pairs = [
        (int(idx_1) if idx_1 != -1 else None, int(idx_2) if idx_2 != -1 else None)
        for idx_1, idx_2 in zip(*alignment.indices, strict=True)
    ]

    return pairs


def get_first_protein_chain(model: Model) -> ProteinChain:
    """Extract first protein chain from structure.

    Args:
        model: Biopython Model

    Returns:
        First protein chain found in model

    Raises:
        ValueError: If no protein chain found
    """
    for chain in model:
        try:
            return create_chain(chain)
        except ValueError:
            continue

    raise ValueError("No protein chain found in structure")


def get_chain(model: Model, chain_id: str | None = None) -> ProteinChain:
    """Get chain by ID or first protein chain.

    Args:
        model: Biopython Model
        chain_id: Chain identifier (e.g. "A", "B"). If None, returns first protein chain.

    Returns:
        Requested chain

    Raises:
        ValueError: If chain_id specified but not found, or chain is not protein
    """
    if chain_id is None:
        return get_first_protein_chain(model)

    for chain in model:
        try:
            protein_chain = create_chain(chain)
            if protein_chain.chain_id == chain_id:
                return protein_chain
        except ValueError:
            continue

    raise ValueError(f"Chain '{chain_id}' is not a protein chain or not found")


def get_all_protein_chains(model: Model) -> list[ProteinChain]:
    """Get all protein chains from model.

    Args:
        model: Biopython Model

    Returns:
        List of protein chains

    Raises:
        ValueError: If no protein chains found
    """
    chains: list[ProteinChain] = []
    for chain in model:
        try:
            chains.append(create_chain(chain))
        except ValueError:
            continue

    if len(chains) == 0:
        raise ValueError("No protein chains found in model")

    return chains


def load_structure(path: str | Path) -> Structure:
    """Load a protein structure from PDB or mmCIF file.

    Args:
        path: Path to structure file (PDB or mmCIF format)

    Returns:
        Loaded Biopython Structure object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If structure has no models
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = file_path.suffix.lower()
    parser: PDBParser | MMCIFParser
    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)  # type: ignore[no-untyped-call]
    else:
        parser = PDBParser(QUIET=True)  # type: ignore[no-untyped-call]

    structure = parser.get_structure("structure", str(file_path))  # type: ignore[no-untyped-call]

    if len(structure) == 0:
        raise ValueError(f"Structure has no models: {path}")

    return cast(Structure, structure)


def write_structure(structure: Structure, path: str | Path) -> None:
    """Write structure to file with format auto-detection.

    Args:
        structure: Biopython Structure to write
        path: Output file path (.pdb or .cif extension)
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    io: PDBIO | MMCIFIO
    if suffix == ".cif":
        io = MMCIFIO()  # type: ignore[no-untyped-call]
    else:
        io = PDBIO()  # type: ignore[no-untyped-call]

    io.set_structure(structure)  # type: ignore[no-untyped-call]
    io.save(str(file_path))


def apply_transformation(
    structure: Structure,
    rotation: NDArray[np.floating],
    translation: NDArray[np.floating],
) -> None:
    """Apply rigid-body transformation to all atoms in structure.

    Transforms all atoms in-place using: new_pos = old_pos @ rotation.T + translation

    Args:
        structure: Input Biopython Structure (modified in-place)
        rotation: Rotation matrix, shape (3, 3)
        translation: Translation vector, shape (3,)
    """
    for atom in structure.get_atoms():  # type: ignore[no-untyped-call]
        atom = cast(Atom, atom)
        pos = atom.get_coord()
        new_pos = pos @ rotation.T + translation
        atom.set_coord(new_pos.astype(float))


def generate_conflict_free_chain_map(
    structure: Structure,
    chain_mapping: dict[str, str],
) -> dict[str, str]:
    """Generate chain rename map that avoids conflicts with unaligned chains.

    For quaternary alignment with chain renaming, this ensures that renaming
    aligned chains doesn't create duplicate chain names with unaligned chains.
    Uses swap strategy: if renaming mobile chain A to B conflicts with existing
    unaligned chain B, adds B->A swap to the map.

    Args:
        structure: Mobile structure
        chain_mapping: Dictionary with chain mappings (fixed to mobile) from alignment

    Returns:
        Complete rename mapping with conflict resolution via swaps
    """
    all_mobile_chains = {chain.id for chain in structure[0]}
    aligned_mobile_chains = {mobile_name for mobile_name in chain_mapping.values()}

    chain_rename_map: dict[str, str] = {}

    for fixed_name, mobile_name in chain_mapping.items():
        if fixed_name == mobile_name:
            continue

        chain_rename_map[mobile_name] = fixed_name

        if fixed_name in all_mobile_chains and fixed_name not in aligned_mobile_chains:
            chain_rename_map[fixed_name] = mobile_name

    return chain_rename_map


def rename_chains(structure: Structure, chain_map: dict[str, str]) -> None:
    """Rename chains using temporary names to avoid collisions.

    Args:
        structure: Biopython Structure (modified in-place)
        chain_map: Mapping of old -> new chain IDs
    """
    if not chain_map:
        return

    # Filter out identity renames
    filtered_map = {old: new for old, new in chain_map.items() if old != new}
    if not filtered_map:
        return

    # Step 1: Rename to temporary names to avoid collisions
    temp_map: dict[str, str] = {}
    for model_index, model in enumerate(structure):
        for chain_index, chain in enumerate(model):
            if chain.id in filtered_map:
                temp_name = f"__CHN_{model_index}_{chain_index}__"
                temp_map[temp_name] = filtered_map[chain.id]
                chain.id = temp_name

    # Step 2: Rename from temporary to final names
    for model in structure:
        for chain in model:
            if chain.id in temp_map:
                chain.id = temp_map[chain.id]
