#!/usr/bin/env python3
"""Compare ProteinChain extraction between Biopython and Gemmi."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser  # type: ignore[import-not-found]

from pyprotalign.bio_utils import create_chain as create_chain_bio
from pyprotalign.gemmi_utils import create_chain as create_chain_gemmi
from pyprotalign.gemmi_utils import load_structure


def _pick_default_cif(struct_dir: Path) -> Path:
    cifs = sorted(struct_dir.glob("*.cif"))
    if not cifs:
        raise FileNotFoundError(f"No .cif files found in: {struct_dir}")
    return cifs[0]


def _summarize_chain(label: str, chain_id: str, chain) -> str:
    valid_ca = int(np.sum(~np.isnan(chain.coords[:, 0])))
    nan_b = int(np.sum(np.isnan(chain.b_factors)))
    nan_occ = int(np.sum(np.isnan(chain.occupancies)))
    return (
        f"{label} chain {chain_id}: len={len(chain.sequence)}, "
        f"coords={chain.coords.shape}, valid_ca={valid_ca}, "
        f"nan_b={nan_b}, nan_occ={nan_occ}"
    )


def _compare_numeric(label: str, a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape:
        return f"{label}: shape_mismatch {a.shape} vs {b.shape}"

    finite_mask = np.isfinite(a) & np.isfinite(b)
    total = int(np.prod(a.shape))
    comparable = int(np.sum(finite_mask))
    allclose = bool(np.allclose(a, b, equal_nan=True))

    if comparable == 0:
        return f"{label}: no_comparable_values allclose={allclose}"

    diffs = np.abs(a[finite_mask] - b[finite_mask])
    max_diff = float(np.max(diffs)) if diffs.size else 0.0
    return f"{label}: allclose={allclose}, comparable={comparable}/{total}, max_abs_diff={max_diff:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Biopython vs Gemmi chain extraction.")
    parser.add_argument(
        "cif",
        nargs="?",
        default=None,
        help="Path to CIF file (defaults to first .cif in test_structures)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    struct_dir = base_dir / "test_structures"

    cif_path = Path(args.cif) if args.cif else _pick_default_cif(struct_dir)
    if not cif_path.is_absolute():
        cif_path = (base_dir / cif_path).resolve()

    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    print(f"Using CIF: {cif_path}")

    # Gemmi parsing
    gemmi_structure = load_structure(cif_path)
    gemmi_model = gemmi_structure[0]
    gemmi_chains: dict[str, object] = {}
    for chain in gemmi_model:
        try:
            gemmi_chain = create_chain_gemmi(chain)
        except ValueError:
            continue
        gemmi_chains[gemmi_chain.chain_id] = gemmi_chain

    # Biopython parsing
    parser = MMCIFParser(QUIET=True)
    bio_structure = parser.get_structure("bio", str(cif_path))
    bio_model = bio_structure[0]
    bio_chains: dict[str, object] = {}
    for chain in bio_model:
        try:
            bio_chain = create_chain_bio(chain)
        except ValueError:
            continue
        bio_chains[bio_chain.chain_id] = bio_chain

    gemmi_ids = set(gemmi_chains)
    bio_ids = set(bio_chains)

    print(f"Gemmi protein chains (sorted): {sorted(gemmi_ids)}")
    print(f"Biopython protein chains (sorted): {sorted(bio_ids)}")
    print(f"Gemmi protein chains (unordered): {gemmi_ids}")
    print(f"Biopython protein chains (unordered): {bio_ids}")

    only_gemmi = sorted(gemmi_ids - bio_ids)
    only_bio = sorted(bio_ids - gemmi_ids)
    if only_gemmi:
        print(f"Only in gemmi: {only_gemmi}")
    if only_bio:
        print(f"Only in biopython: {only_bio}")

    for chain_id in sorted(gemmi_ids & bio_ids):
        g = gemmi_chains[chain_id]
        b = bio_chains[chain_id]
        print(_summarize_chain("Gemmi", chain_id, g))
        print(_summarize_chain("Bio  ", chain_id, b))
        seq_equal = g.sequence == b.sequence
        len_equal = len(g.sequence) == len(b.sequence)
        print(f"  sequence_equal={seq_equal}, length_equal={len_equal}")
        print(f"  {_compare_numeric('coords', g.coords, b.coords)}")
        print(f"  {_compare_numeric('b_factors', g.b_factors, b.b_factors)}")
        print(f"  {_compare_numeric('occupancies', g.occupancies, b.occupancies)}")


if __name__ == "__main__":
    main()
