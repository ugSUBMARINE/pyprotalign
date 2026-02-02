"""Command-line interface for pyprotalign."""

import argparse
import sys
from pathlib import Path

import gemmi
import numpy as np

from . import __version__
from .alignment import align_multi_chain, align_quaternary, align_sequences
from .io import load_structure, write_structure
from .kabsch import calculate_rmsd, superpose
from .refine import iterative_superpose
from .selection import extract_ca_atoms, extract_sequence, get_all_protein_chains, get_chain
from .transform import apply_transformation, generate_conflict_free_chain_map, rename_chains


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
        if args.verbose:
            print("=== Quaternary Alignment ===")
        fixed_coords, mobile_coords, chain_pairs = align_quaternary(
            fixed_st,
            mobile_st,
            distance_threshold=args.distance_threshold,
            seed_fixed_chain=args.fixed_chain,
            seed_mobile_chain=args.mobile_chain,
            refine=args.refine,
            cutoff_factor=args.cutoff,
            max_cycles=args.cycles,
            verbose=args.verbose,
        )

        # Print chain pairing information
        print("Quaternary alignment:")
        fixed_chains = get_all_protein_chains(fixed_st)
        fixed_chain_names = {chain.name for chain in fixed_chains}
        paired_fixed = {pair[0] for pair in chain_pairs}

        for fixed_name, mobile_name in chain_pairs:
            print(f"  {fixed_name} → {mobile_name} (matched)")

        # Show unmatched chains
        unmatched_fixed = fixed_chain_names - paired_fixed
        for fixed_name in sorted(unmatched_fixed):
            print(f"  {fixed_name} → - (no match)")

        print(f"Aligned: {len(fixed_coords)} CA pairs across {len(chain_pairs)} chain pairs")
        info_str = f"{len(chain_pairs)} chain pairs"

    elif args.global_mode:
        # Global multi-chain alignment
        fixed_coords, mobile_coords, chain_ids = align_multi_chain(fixed_st, mobile_st)
        # Count chains and pairs
        unique_chains = sorted(set(chain_ids))
        print(f"Chains: {', '.join(unique_chains)}")
        print(f"Aligned: {len(fixed_coords)} CA atom pairs across {len(unique_chains)} chains")
        info_str = f"{len(unique_chains)} chains"
    else:
        # Single-chain alignment
        fixed_chain = get_chain(fixed_st, args.fixed_chain)
        mobile_chain = get_chain(mobile_st, args.mobile_chain)

        # Extract sequences
        fixed_seq = extract_sequence(fixed_chain)
        mobile_seq = extract_sequence(mobile_chain)

        print(f"Fixed:  chain {fixed_chain.name}, {len(fixed_seq)} residues")
        print(f"Mobile: chain {mobile_chain.name}, {len(mobile_seq)} residues")

        # Align sequences
        pairs = align_sequences(fixed_seq, mobile_seq)

        # Extract CA atoms
        fixed_cas = extract_ca_atoms(fixed_chain)
        mobile_cas = extract_ca_atoms(mobile_chain)

        # Filter aligned pairs (exclude gaps)
        fixed_indices = []
        mobile_indices = []
        for fix_idx, mob_idx in pairs:
            if fix_idx is not None and mob_idx is not None:
                # Check that CA atoms exist
                if fix_idx < len(fixed_cas) and mob_idx < len(mobile_cas):
                    fixed_indices.append(fix_idx)
                    mobile_indices.append(mob_idx)

        if len(fixed_indices) < 3:
            raise ValueError(f"Need at least 3 aligned CA pairs, found {len(fixed_indices)}")

        print(f"Aligned: {len(fixed_indices)} CA atom pairs")

        # Extract coordinates
        fixed_coords = np.array([[fixed_cas[i].pos.x, fixed_cas[i].pos.y, fixed_cas[i].pos.z] for i in fixed_indices])
        mobile_coords = np.array(
            [[mobile_cas[i].pos.x, mobile_cas[i].pos.y, mobile_cas[i].pos.z] for i in mobile_indices]
        )
        info_str = f"{fixed_chain.name} → {mobile_chain.name}"

    # Compute transformation (not needed for quaternary mode, already computed)
    if not args.quaternary:
        if args.refine:
            if args.verbose:
                print("=== Final Refinement ===")
            rotation, translation, mask, rmsd = iterative_superpose(
                fixed_coords, mobile_coords, max_cycles=args.cycles, cutoff_factor=args.cutoff, verbose=args.verbose
            )
            n_used = np.sum(mask)
            n_rejected = len(mask) - n_used
            if not args.verbose:
                print(f"Refinement: {n_used} pairs retained, {n_rejected} rejected")
        else:
            rotation, translation = superpose(fixed_coords, mobile_coords)
            # Calculate RMSD for reporting
            mobile_transformed = mobile_coords @ rotation.T + translation
            rmsd = calculate_rmsd(fixed_coords, mobile_transformed)

        print(f"RMSD: {rmsd:.3f} Å")

        # Apply transformation to entire mobile structure
        apply_transformation(mobile_st, rotation, translation)
    else:
        # Quaternary mode: compute final transformation with optional refinement
        if args.refine:
            if args.verbose:
                print("=== Final Refinement ===")
            rotation, translation, mask, rmsd = iterative_superpose(
                fixed_coords, mobile_coords, max_cycles=args.cycles, cutoff_factor=args.cutoff, verbose=args.verbose
            )
            n_used = np.sum(mask)
            n_rejected = len(mask) - n_used
            if not args.verbose:
                print(f"Refinement: {n_used} CA pairs retained, {n_rejected} rejected")
        else:
            rotation, translation = superpose(fixed_coords, mobile_coords)
            mobile_transformed = mobile_coords @ rotation.T + translation
            rmsd = calculate_rmsd(fixed_coords, mobile_transformed)

        print(f"RMSD: {rmsd:.3f} Å")

        # Apply transformation
        apply_transformation(mobile_st, rotation, translation)

        # Rename mobile chains to match fixed (if requested)
        if args.rename_chains:
            chain_rename_map = generate_conflict_free_chain_map(mobile_st, chain_pairs)
            if args.verbose:
                # Get all mobile chain names for complete reporting
                all_mobile_chains = sorted({chain.name for model in mobile_st for chain in model})
                aligned_renames = {mobile_name: fixed_name for fixed_name, mobile_name in chain_pairs}

                print("Chain renaming:")
                for chain_name in all_mobile_chains:
                    if chain_name in chain_rename_map:
                        new_name = chain_rename_map[chain_name]
                        if chain_name in aligned_renames and aligned_renames[chain_name] == new_name:
                            print(f"  {chain_name} → {new_name} (aligned)")
                        else:
                            print(f"  {chain_name} → {new_name} (unaligned, renamed to avoid conflict)")
                    elif chain_name in aligned_renames:
                        # Identity rename (same name), aligned but not in map
                        print(f"  {chain_name} → {chain_name} (aligned)")
                    else:
                        # Unaligned chain that keeps its name
                        print(f"  {chain_name} → {chain_name} (unaligned)")
            rename_chains(mobile_st, chain_rename_map)

    return mobile_st, rmsd, info_str


def main() -> None:
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

    # Validate mutual exclusivity
    if args.global_mode and args.quaternary:
        parser.error("--global and --quaternary are mutually exclusive")
    if args.global_mode and (args.fixed_chain or args.mobile_chain):
        parser.error("--global cannot be used with --fixed-chain or --mobile-chain")
    if args.rename_chains and not args.quaternary:
        parser.error("--rename-chains can only be used with --quaternary")

    # Determine mode: single or batch
    if len(args.mobile) == 1:
        # Single mode
        try:
            fixed_st = load_structure(args.fixed)
            mobile_st = load_structure(args.mobile[0])

            # Perform alignment
            mobile_st, rmsd, info_str = _align_structures(fixed_st, mobile_st, args)

            # Write output
            write_structure(mobile_st, args.output)
            print(f"Superposed structure written to: {args.output}")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Batch mode
        try:
            # Load fixed structure once
            fixed_st = load_structure(args.fixed)
        except Exception as e:
            print(f"Error loading fixed structure: {e}", file=sys.stderr)
            sys.exit(1)

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
            print(f"Processing {idx}/{num_mobile}: {mobile_path.name}")

            # Skip if mobile file is the same as fixed file
            if mobile_path.resolve() == Path(args.fixed).resolve():
                print("Skipping: same as fixed structure\n")
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

                print(f"Output: {output_file.name}")

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
                print(f"  Error: {e}")

            print()

        # Print summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        successful = sum(1 for r in results if r["status"] == "OK")
        failed = sum(1 for r in results if r["status"] == "FAILED")

        print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")
        print()

        if successful > 0:
            print("Successful alignments:")
            for r in results:
                if r["status"] == "OK":
                    print(f"  {r['file']:40s} RMSD: {r['rmsd']:.3f} Å → {r['output']}")

        if failed > 0:
            print()
            print("Failed alignments:")
            for r in results:
                if r["status"] == "FAILED":
                    print(f"  {r['file']:40s} Error: {r['info']}")

        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
