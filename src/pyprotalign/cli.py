"""Command-line interface for pyprotalign."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gemmi
import numpy as np

from pyprotalign.chain import ProteinChain

from . import __version__
from .alignment import align_globally, align_hungarian, align_quaternary, align_two_chains
from .gemmi_utils import (
    apply_transformation,
    generate_conflict_free_chain_map,
    get_all_protein_chains,
    get_chain,
    load_structure,
    rename_chains,
    write_structure,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
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
        "--filter",
        action="store_true",
        help="Apply filtering of CA atoms based on B-factors and occupancies",
    )
    parser.add_argument(
        "--min-bfac",
        type=float,
        default=-np.inf,
        help="Minimum B-factor threshold - filter CA atoms with B-factors < threshold (default: -inf)",
    )
    parser.add_argument(
        "--max-bfac",
        type=float,
        default=np.inf,
        help="Maximum B-factor threshold - filter CA atoms with B-factors > threshold (default: inf)",
    )
    parser.add_argument(
        "--min-occ",
        type=float,
        default=-np.inf,
        help="Minimum occupancy threshold - filter CA atoms with occupancies < threshold (default: -inf)",
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
        help="Align all protein chains by matching chain IDs or their order in the structure and pooling coordinates",
    )
    parser.add_argument(
        "--by-order",
        action="store_true",
        dest="by_order",
        help="Use the order of the chains in the structure for chain matching",
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
        "--match",
        choices=["hungarian", "greedy"],
        default=None,
        help="Chain matching algorithm for quaternary mode (default: greedy)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (show refinement cycles, chain matching details)",
    )

    args = parser.parse_args()

    if args.filter and (args.min_bfac > args.max_bfac):
        parser.error("--min-bfac cannot be greater than --max-bfac")

    # Validate mutual exclusivity
    if args.global_mode and args.quaternary:
        parser.error("--global and --quaternary are mutually exclusive")
    if args.global_mode and (args.fixed_chain or args.mobile_chain):
        parser.error("--global cannot be used with --fixed-chain or --mobile-chain")
    if args.by_order and not args.global_mode:
        parser.error("--by-order can only be used with --global")
    if args.rename_chains and not args.quaternary:
        parser.error("--rename-chains can only be used with --quaternary")
    if args.match is not None and not args.quaternary:
        parser.error("--match can only be used with --quaternary")

    args.match = args.match or "greedy"

    return args


def _align_two_chains(
    fixed_chain: ProteinChain, mobile_st: gemmi.Structure, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Align two chains and return rotation, translation, RMSD, and number of aligned pairs."""
    try:
        mobile_chain = get_chain(mobile_st[0], args.mobile_chain)
    except ValueError as e:
        raise ValueError(f"Mobile structure has no matching protein chain: {e}") from e

    logger.info("Fixed:  chain %s, %d residues", fixed_chain.chain_id, len(fixed_chain.sequence))
    logger.info("Mobile: chain %s, %d residues", mobile_chain.chain_id, len(mobile_chain.sequence))

    rotation, translation, rmsd, num_aligned = align_two_chains(
        fixed_chain,
        mobile_chain,
        refine=args.refine,
        max_cycles=args.cycles,
        cutoff_factor=args.cutoff,
        filter=args.filter,
        min_bfactor=args.min_bfac,
        max_bfactor=args.max_bfac,
        min_occ=args.min_occ,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("\n-- Summary --\n")

    return rotation, translation, rmsd, num_aligned


def _align_globally(
    fixed_chains_map: dict[str, ProteinChain], mobile_st: gemmi.Structure, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Align all chains globally and return rotation, translation, RMSD, and number of aligned pairs."""
    try:
        mobile_chains = get_all_protein_chains(mobile_st[0])
    except ValueError as e:
        raise ValueError(f"Mobile structure has no protein chains: {e}") from e
    mobile_chains_map = {chain.chain_id: chain for chain in mobile_chains}

    rotation, translation, rmsd, num_aligned, chain_mapping = align_globally(
        fixed_chains_map,
        mobile_chains_map,
        by_chain_id=not args.by_order,
        refine=args.refine,
        max_cycles=args.cycles,
        cutoff_factor=args.cutoff,
        filter=args.filter,
        min_bfactor=args.min_bfac,
        max_bfactor=args.max_bfac,
        min_occ=args.min_occ,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("\n-- Summary --\n")

    if args.by_order:
        matches = ", ".join([f"{chn_id_1} → {chn_id_2}" for chn_id_1, chn_id_2 in chain_mapping.items()])
        logger.info(
            "Aligned CA pairs across %d chain pairings (fixed → mobile): %s",
            len(chain_mapping),
            matches,
        )
    else:
        logger.info("Aligned CA pairs across %d chains: %s", len(chain_mapping), ", ".join(chain_mapping.keys()))

    return rotation, translation, rmsd, num_aligned


def _align_quaternary(
    fixed_chains: list[ProteinChain], mobile_st: gemmi.Structure, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, str]]:
    """Perform quaternary alignment and return rotation, translation, RMSD, and number of aligned pairs."""
    try:
        mobile_chains = get_all_protein_chains(mobile_st[0])
    except ValueError as e:
        raise ValueError(f"Mobile structure has no protein chains: {e}") from e

    align_fn = align_hungarian if args.match == "hungarian" else align_quaternary
    rotation, translation, rmsd, num_aligned, chain_mapping = align_fn(
        fixed_chains,
        mobile_chains,
        fixed_seed=args.fixed_chain,
        mobile_seed=args.mobile_chain,
        distance_threshold=args.distance_threshold,
        refine=args.refine,
        cutoff_factor=args.cutoff,
        max_cycles=args.cycles,
        filter=args.filter,
        min_bfactor=args.min_bfac,
        max_bfactor=args.max_bfac,
        min_occ=args.min_occ,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("\n-- Summary --\n")

    matches = ", ".join([f"{chn_id_1} → {chn_id_2}" for chn_id_1, chn_id_2 in chain_mapping.items()])
    logger.info(
        "Aligned CA pairs across %d chain pairings (fixed → mobile): %s",
        len(chain_mapping),
        matches,
    )

    return rotation, translation, rmsd, num_aligned, chain_mapping


def main() -> int:
    """Entry point for the protalign CLI."""
    args = _parse_args()
    _configure_logging(args.verbose)

    # Parse suffix from --output (strip extension if present)
    output_path = Path(args.output)
    if output_path.suffix in [".cif", ".pdb"]:
        suffix = output_path.stem
        output_ext = output_path.suffix
    else:
        suffix = args.output
        output_ext = ".cif"

    # Log headings
    if args.global_mode:
        logger.info("=== Global Multi-Chain Alignment ===\n")
    elif args.quaternary:
        logger.info("=== Quaternary Alignment ===\n")
    else:
        logger.info("=== Single-Chain Alignment ===\n")

    # Load fixed structure and extract protein chains
    try:
        fixed_path = Path(args.fixed)
        fixed_st = load_structure(fixed_path)
        logger.info("## Fixed structure loaded: %s", fixed_path.name)
        num_models = len(fixed_st)
        if num_models > 1:
            logger.warning("Structure contains %d models; only the first model will be used as target.", num_models)

        # Extract protein chains from fixed structure
        fixed_chains = get_all_protein_chains(fixed_st[0])
        if not fixed_chains:
            raise ValueError("No protein chains found in fixed structure")
        fixed_chains_map = {chain.chain_id: chain for chain in fixed_chains}

        if args.fixed_chain and args.fixed_chain not in fixed_chains_map:
            raise ValueError(f"Fixed chain '{args.fixed_chain}' not found in structure")

        fixed_chain = fixed_chains_map[args.fixed_chain] if args.fixed_chain else fixed_chains[0]

        if args.global_mode:
            logger.info("")
        elif args.quaternary:
            logger.info("Using chain '%s' as seed for quaternary alignment.\n", fixed_chain.chain_id)
        else:
            logger.info("Using chain '%s' as target for superposition.\n", fixed_chain.chain_id)

    except Exception as e:
        logger.error("Error: %s\n", e)
        return 1

    ok: list[bool] = []
    num_mobile = len(args.mobile)
    for idx, mobile in enumerate(args.mobile, 1):
        mobile_path = Path(mobile)
        if num_mobile > 1:
            logger.info("## Processing %d/%d: %s", idx, num_mobile, mobile_path.name)
        else:
            logger.info("## Processing: %s", mobile_path.name)

        try:
            mobile_st = load_structure(mobile_path)
            num_models = len(mobile_st)
            if num_models > 1:
                logger.warning(
                    "Structure contains %d models; only the first model will be used.\n"
                    "The transformation will be applied to the whole structure.",
                    num_models,
                )

            # Perform the different types of alignment
            if args.global_mode:
                rotation, translation, rmsd, num_aligned = _align_globally(fixed_chains_map, mobile_st, args)
            elif args.quaternary:
                rotation, translation, rmsd, num_aligned, chain_mapping = _align_quaternary(
                    fixed_chains, mobile_st, args
                )
            else:
                rotation, translation, rmsd, num_aligned = _align_two_chains(fixed_chain, mobile_st, args)

            # Apply transformation to mobile structure
            logger.info("Aligned %d CA pairs with RMSD %.3f Å", num_aligned, rmsd)
            apply_transformation(mobile_st, rotation, translation)

            # Rename the chains in mobile structure after quaternary alignment
            if args.rename_chains:
                chain_rename_map = generate_conflict_free_chain_map(mobile_st, chain_mapping)
                logger.info(
                    "Renaming chains in mobile structure: %s",
                    ", ".join([f"{chn_id_1} → {chn_id_2}" for chn_id_1, chn_id_2 in chain_rename_map.items()]),
                )
                if logger.isEnabledFor(logging.DEBUG):
                    # Get all mobile chain names for complete reporting
                    logger.debug("Complete mapping:")
                    all_mobile_chains = sorted({chain.name for chain in mobile_st[0]})
                    aligned_renames = {mobile_name: fixed_name for fixed_name, mobile_name in chain_mapping.items()}
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
                    logger.debug("")

                rename_chains(mobile_st, chain_rename_map)

            # Write output
            if num_mobile == 1:
                output_file = Path(args.output)
                if output_file.suffix.lower() not in [".cif", ".pdb"]:
                    output_file = output_file.with_suffix(".cif")
            else:
                output_file = Path.cwd() / f"{mobile_path.stem}_{suffix}{output_ext}"
            write_structure(mobile_st, str(output_file))
            logger.info("Superposed structure written to: %s\n", output_file)

            ok.append(True)

        except Exception as e:
            logger.error("Error aligning `%s`: %s\n", mobile, e)
            ok.append(False)

    return 0 if all(ok) else 1


if __name__ == "__main__":
    main()
