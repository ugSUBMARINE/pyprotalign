"""Command-line interface for pyprotalign."""

import argparse
import logging
from pathlib import Path

import gemmi
import numpy as np

from . import __version__
from .alignment import align_multi_chain, align_quaternary, align_sequences
from .io import load_structure, write_structure
from .kabsch import calculate_rmsd, superpose
from .refine import iterative_superpose
from .selection import (
    extract_ca_atoms_by_residue,
    extract_sequence,
    filter_ca_pairs_by_quality,
    get_all_protein_chains,
    get_chain,
)
from .transform import apply_transformation, generate_conflict_free_chain_map, rename_chains

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _align_structures(
    fixed_st: gemmi.Structure,
    mobile_st: gemmi.Structure,
    args: argparse.Namespace,
) -> tuple[gemmi.Structure, float, str]:
    """Perform structure alignment and return transformed structure, RMSD, and info string.

    Args:
        fixed_st: Fixed structure
        mobile_st: Mobile structure - will be modified in place
        args: Parsed command-line arguments

    Returns:
        Tuple of (transformed mobile structure, RMSD, info string for reporting)
    """
    # Choose alignment mode
    if args.quaternary:
        # Quaternary alignment with smart chain matching
        logger.debug("=== Quaternary Alignment ===")
        fixed_coords, mobile_coords, chain_pairs = align_quaternary(
            fixed_st,
            mobile_st,
            distance_threshold=args.distance_threshold,
            seed_fixed_chain=args.fixed_chain,
            seed_mobile_chain=args.mobile_chain,
            refine=args.refine,
            cutoff_factor=args.cutoff,
            max_cycles=args.cycles,
            min_plddt=args.plddt,
            max_bfactor=args.bfactor,
        )

        # Log chain pairing information
        logger.info("Quaternary alignment:")
        fixed_chains = get_all_protein_chains(fixed_st)
        fixed_chain_names = {chain.name for chain in fixed_chains}
        paired_fixed = {pair[0] for pair in chain_pairs}

        for fixed_name, mobile_name in chain_pairs:
            logger.info("  %s → %s (matched)", fixed_name, mobile_name)

        # Show unmatched chains
        unmatched_fixed = fixed_chain_names - paired_fixed
        for fixed_name in sorted(unmatched_fixed):
            logger.info("  %s → - (no match)", fixed_name)

        logger.info("Aligned: %d CA pairs across %d chain pairs", len(fixed_coords), len(chain_pairs))
        info_str = f"{len(chain_pairs)} chain pairs"

    elif args.global_mode:
        # Global multi-chain alignment
        fixed_coords, mobile_coords, chain_ids = align_multi_chain(
            fixed_st, mobile_st, min_plddt=args.plddt, max_bfactor=args.bfactor
        )
        # Count chains and pairs
        unique_chains = sorted(set(chain_ids))
        logger.info("Chains: %s", ", ".join(unique_chains))
        logger.info("Aligned: %d CA atom pairs across %d chains", len(fixed_coords), len(unique_chains))
        info_str = f"{len(unique_chains)} chains"
    else:
        # Single-chain alignment
        fixed_chain = get_chain(fixed_st, args.fixed_chain)
        mobile_chain = get_chain(mobile_st, args.mobile_chain)

        # Extract sequences
        fixed_seq = extract_sequence(fixed_chain)
        mobile_seq = extract_sequence(mobile_chain)

        logger.info("Fixed:  chain %s, %d residues", fixed_chain.name, len(fixed_seq))
        logger.info("Mobile: chain %s, %d residues", mobile_chain.name, len(mobile_seq))

        # Align sequences
        pairs = align_sequences(fixed_seq, mobile_seq)

        # Extract CA atoms
        fixed_cas = extract_ca_atoms_by_residue(fixed_chain)
        mobile_cas = extract_ca_atoms_by_residue(mobile_chain)

        # Filter aligned pairs (exclude gaps)
        fixed_indices = []
        mobile_indices = []
        for fix_idx, mob_idx in pairs:
            if fix_idx is not None and mob_idx is not None:
                # Check that CA atoms exist
                if fix_idx < len(fixed_cas) and mob_idx < len(mobile_cas):
                    if fixed_cas[fix_idx] is None or mobile_cas[mob_idx] is None:
                        continue
                    fixed_indices.append(fix_idx)
                    mobile_indices.append(mob_idx)

        if len(fixed_indices) < 3:
            raise ValueError(f"Need at least 3 aligned CA pairs, found {len(fixed_indices)}")

        logger.info("Aligned: %d CA atom pairs", len(fixed_indices))

        # Apply quality filtering if requested
        if args.plddt is not None or args.bfactor is not None:
            # Build aligned CA atom lists
            aligned_fixed_cas = [fixed_cas[i] for i in fixed_indices]
            aligned_mobile_cas = [mobile_cas[i] for i in mobile_indices]

            # Filter by quality
            quality_mask = filter_ca_pairs_by_quality(
                aligned_fixed_cas, aligned_mobile_cas, min_plddt=args.plddt, max_bfactor=args.bfactor
            )

            # Apply mask to indices
            filtered_fixed_indices = [fixed_indices[i] for i in range(len(fixed_indices)) if quality_mask[i]]
            filtered_mobile_indices = [mobile_indices[i] for i in range(len(mobile_indices)) if quality_mask[i]]

            n_filtered = len(fixed_indices) - len(filtered_fixed_indices)
            filter_type = "pLDDT" if args.plddt is not None else "B-factor"
            threshold = args.plddt if args.plddt is not None else args.bfactor
            logger.info(
                "Quality filter (%s %.1f): %d pairs filtered, %d retained",
                filter_type,
                threshold,
                n_filtered,
                len(filtered_fixed_indices),
            )

            fixed_indices = filtered_fixed_indices
            mobile_indices = filtered_mobile_indices

            if len(fixed_indices) < 3:
                raise ValueError(f"Need at least 3 CA pairs after quality filtering, found {len(fixed_indices)}")

        # Extract coordinates
        fixed_coords_list = []
        mobile_coords_list = []
        for fix_idx, mob_idx in zip(fixed_indices, mobile_indices, strict=True):
            fixed_ca = fixed_cas[fix_idx]
            mobile_ca = mobile_cas[mob_idx]
            if fixed_ca is None or mobile_ca is None:
                continue
            fixed_coords_list.append([fixed_ca.pos.x, fixed_ca.pos.y, fixed_ca.pos.z])
            mobile_coords_list.append([mobile_ca.pos.x, mobile_ca.pos.y, mobile_ca.pos.z])

        fixed_coords = np.array(fixed_coords_list)
        mobile_coords = np.array(mobile_coords_list)
        info_str = f"{fixed_chain.name} → {mobile_chain.name}"

    # Compute transformation (not needed for quaternary mode, already computed)
    if not args.quaternary:
        if args.refine:
            logger.debug("=== Final Refinement ===")
            rotation, translation, mask, rmsd = iterative_superpose(
                fixed_coords, mobile_coords, max_cycles=args.cycles, cutoff_factor=args.cutoff
            )
            n_used = np.sum(mask)
            n_rejected = len(mask) - n_used
            logger.info("Refinement: %d pairs retained, %d rejected", n_used, n_rejected)
        else:
            rotation, translation = superpose(fixed_coords, mobile_coords)
            # Calculate RMSD for reporting
            mobile_transformed = mobile_coords @ rotation.T + translation
            rmsd = calculate_rmsd(fixed_coords, mobile_transformed)

        logger.info("RMSD: %.3f Å", rmsd)

        # Apply transformation to entire mobile structure
        apply_transformation(mobile_st, rotation, translation)
    else:
        # Quaternary mode: compute final transformation with optional refinement
        if args.refine:
            logger.debug("=== Final Refinement ===")
            rotation, translation, mask, rmsd = iterative_superpose(
                fixed_coords, mobile_coords, max_cycles=args.cycles, cutoff_factor=args.cutoff
            )
            n_used = np.sum(mask)
            n_rejected = len(mask) - n_used
            logger.info("Refinement: %d CA pairs retained, %d rejected", n_used, n_rejected)
        else:
            rotation, translation = superpose(fixed_coords, mobile_coords)
            mobile_transformed = mobile_coords @ rotation.T + translation
            rmsd = calculate_rmsd(fixed_coords, mobile_transformed)

        logger.info("RMSD: %.3f Å", rmsd)

        # Apply transformation
        apply_transformation(mobile_st, rotation, translation)

        # Rename mobile chains to match fixed (if requested)
        if args.rename_chains:
            chain_rename_map = generate_conflict_free_chain_map(mobile_st, chain_pairs)
            # Get all mobile chain names for complete reporting
            all_mobile_chains = sorted({chain.name for model in mobile_st for chain in model})
            aligned_renames = {mobile_name: fixed_name for fixed_name, mobile_name in chain_pairs}

            logger.debug("Chain renaming:")
            for chain_name in all_mobile_chains:
                if chain_name in chain_rename_map:
                    new_name = chain_rename_map[chain_name]
                    if chain_name in aligned_renames and aligned_renames[chain_name] == new_name:
                        logger.debug("  %s → %s (aligned)", chain_name, new_name)
                    else:
                        logger.debug("  %s → %s (unaligned, renamed to avoid conflict)", chain_name, new_name)
                elif chain_name in aligned_renames:
                    # Identity rename (same name), aligned but not in map
                    logger.debug("  %s → %s (aligned)", chain_name, chain_name)
                else:
                    # Unaligned chain that keeps its name
                    logger.debug("  %s → %s (unaligned)", chain_name, chain_name)
            rename_chains(mobile_st, chain_rename_map)

    return mobile_st, rmsd, info_str


def _gemmi_align_structures(
    fixed_st: gemmi.Structure,
    mobile_st: gemmi.Structure,
    args: argparse.Namespace,
) -> tuple[gemmi.Structure, float, str]:
    """Perform simple one-chain-to-one-chain alignment using gemmi's built-in function.

    Args:
        fixed_st: Fixed structure
        mobile_st: Mobile structure - will be modified in place
        args: Parsed command-line arguments

    Returns:
        Tuple of (transformed mobile structure, RMSD, info string for reporting)

    Note:
        Internal sequence alignment yields much smaller numbers of CA pairs ('sup.count')
        than the alignment method used in the custom functions. Reasons are unclear.
    """
    # get chains, default to first chain if not specified
    fixed_chain = fixed_st[0][0] if not args.fixed_chain else fixed_st[0][args.fixed_chain]
    mobile_chain = mobile_st[0][0] if not args.mobile_chain else mobile_st[0][args.mobile_chain]

    fixed_pol = fixed_chain.get_polymer()
    mobile_pol = mobile_chain.get_polymer()
    ptype = fixed_pol.check_polymer_type()

    logger.info("Using gemmi built-in superposition")
    logger.info("Fixed:  chain %s, %d residues", fixed_chain.name, len(fixed_pol))
    logger.info("Mobile: chain %s, %d residues", mobile_chain.name, len(mobile_pol))

    sup = gemmi.calculate_superposition(
        fixed_pol,
        mobile_pol,
        ptype,
        sel=gemmi.SupSelect.CaP,
        trim_cycles=args.cycles if args.refine else 0,
        trim_cutoff=args.cutoff,
    )

    logger.info("Aligned: %d CA atom pairs", sup.count)
    logger.info("RMSD: %.3f Å", sup.rmsd)

    for model in mobile_st:
        model.transform_pos_and_adp(sup.transform)

    return mobile_st, sup.rmsd, "using gemmi superposition"


def main() -> int:
    """Entry point for the protalign CLI."""
    parser = argparse.ArgumentParser(
        prog="protalign",
        description="Protein structure superposition tool",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "fixed",
        help="Fixed structure file (PDB or mmCIF)",
    )
    parser.add_argument(
        "mobile",
        nargs="+",
        help="Mobile structure file(s) (PDB or mmCIF). If multiple files provided, batch mode is activated.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="superposed.cif",
        help="Output file (single mode) or suffix (batch mode) (default: superposed.cif)",
    )
    parser.add_argument(
        "--fixed-chain",
        type=str,
        default=None,
        help="Chain ID for fixed structure (e.g., A). Also used as 'seed' chain in quaternary mode. "
        "If not specified, uses first protein chain.",
    )
    parser.add_argument(
        "--mobile-chain",
        type=str,
        default=None,
        help="Chain ID for mobile structure (e.g., A). Also used as 'seed' chain in quaternary mode. "
        "If not specified, uses first protein chain.",
    )
    parser.add_argument(
        "--plddt",
        type=float,
        default=None,
        help="Minimum pLDDT threshold - filter CA atoms with pLDDT < threshold (mutually exclusive with --bfactor)",
    )
    parser.add_argument(
        "--bfactor",
        type=float,
        default=None,
        help="Maximum B-factor threshold - filter CA atoms with B-factor > threshold (mutually exclusive with --plddt)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Use iterative refinement to reject outliers",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Maximum refinement cycles (default: 5)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=2.0,
        help="Outlier rejection cutoff (distance > cutoff * RMSD) (default: 2.0)",
    )
    parser.add_argument(
        "--global",
        action="store_true",
        dest="global_mode",
        help="Align all protein chains by matching chain IDs (A-A, B-B, etc.) and pooling coordinates",
    )
    parser.add_argument(
        "--quaternary",
        action="store_true",
        help="Quaternary alignment: match chains by proximity, rename to match fixed",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=8.0,
        help="Distance threshold (Å) for chain matching in quaternary mode (default: 8.0)",
    )
    parser.add_argument(
        "--rename-chains",
        action="store_true",
        help="Rename mobile chains to match fixed (only with --quaternary)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (show refinement cycles, chain matching details)",
    )

    args = parser.parse_args()
    _configure_logging(args.verbose)

    # Validate mutual exclusivity
    if args.global_mode and args.quaternary:
        parser.error("--global and --quaternary are mutually exclusive")
    if args.global_mode and (args.fixed_chain or args.mobile_chain):
        parser.error("--global cannot be used with --fixed-chain or --mobile-chain")
    if args.rename_chains and not args.quaternary:
        parser.error("--rename-chains can only be used with --quaternary")
    if args.plddt is not None and args.bfactor is not None:
        parser.error("--plddt and --bfactor are mutually exclusive")

    # Default alignment mode: one mobile chain to one fixed chain, no filtering
    # In this cace, use the alignment function provided by 'gemmi'
    default_align_mode = all(
        [
            args.plddt is None,
            args.bfactor is None,
            not args.global_mode,
            not args.quaternary,
            # always use custom alignment for now
            # there are issues with gemmi alignment
            False,
        ]
    )

    # Determine mode: single or batch
    if len(args.mobile) == 1:
        # Single mode
        try:
            fixed_st = load_structure(args.fixed)
            mobile_st = load_structure(args.mobile[0])

            # Perform alignment
            if default_align_mode:
                mobile_st, rmsd, info_str = _gemmi_align_structures(fixed_st, mobile_st, args)
            else:
                mobile_st, rmsd, info_str = _align_structures(fixed_st, mobile_st, args)

            # Write output
            write_structure(mobile_st, args.output)
            logger.info("Superposed structure written to: %s", args.output)

            return 0

        except Exception as e:
            logger.error("Error: %s", e)
            return 1
    else:
        # Batch mode
        try:
            # Load fixed structure once
            fixed_st = load_structure(args.fixed)
        except Exception as e:
            logger.error("Error loading fixed structure: %s", e)
            return 1

        # Parse suffix from --output (strip extension if present)
        output_path = Path(args.output)
        if output_path.suffix in [".cif", ".pdb"]:
            suffix = output_path.stem
            output_ext = output_path.suffix
        else:
            suffix = args.output
            output_ext = ".cif"

        # Process each mobile file
        results = []
        num_mobile = len(args.mobile)
        for idx, mobile_file in enumerate(args.mobile, 1):
            mobile_path = Path(mobile_file)
            logger.info("Processing %d/%d: %s", idx, num_mobile, mobile_path.name)

            # Skip if mobile file is the same as fixed file
            if mobile_path.resolve() == Path(args.fixed).resolve():
                logger.info("Skipping: same as fixed structure")
                logger.info("")
                continue

            try:
                # Load mobile structure
                mobile_st = load_structure(str(mobile_path))

                # Perform alignment
                mobile_st, rmsd, info_str = _align_structures(fixed_st, mobile_st, args)

                # Write output
                output_file = Path.cwd() / f"{mobile_path.stem}_{suffix}{output_ext}"
                write_structure(mobile_st, str(output_file))

                results.append(
                    {
                        "file": mobile_path.name,
                        "status": "OK",
                        "rmsd": rmsd,
                        "info": info_str,
                        "output": output_file.name,
                    }
                )

                logger.info("Output: %s", output_file.name)

            except Exception as e:
                results.append(
                    {
                        "file": mobile_path.name,
                        "status": "FAILED",
                        "rmsd": None,
                        "info": str(e),
                        "output": None,
                    }
                )
                logger.error("  Error: %s", e)

            logger.info("")

        # Log summary
        logger.info("%s", "=" * 80)
        logger.info("SUMMARY")
        logger.info("%s", "=" * 80)
        successful = sum(1 for r in results if r["status"] == "OK")
        failed = sum(1 for r in results if r["status"] == "FAILED")

        logger.info("Total: %d | Successful: %d | Failed: %d", len(results), successful, failed)
        logger.info("")

        if successful > 0:
            logger.info("Successful alignments:")
            for r in results:
                if r["status"] == "OK":
                    logger.info("  %-40s RMSD: %.3f Å → %s", r["file"], r["rmsd"], r["output"])

        if failed > 0:
            logger.info("")
            logger.info("Failed alignments:")
            for r in results:
                if r["status"] == "FAILED":
                    logger.info("  %-40s Error: %s", r["file"], r["info"])

        return 0 if failed == 0 else 1


if __name__ == "__main__":
    main()
