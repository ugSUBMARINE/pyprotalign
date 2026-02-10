"""All-to-all chain alignment CLI tool."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from . import __version__
from .alignment import align_two_chains
from .gemmi_utils import align_sequences, get_all_protein_chains, load_structure

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="protalign-all2all",
        description="All-to-all chain alignment across one or more structures",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "structure",
        nargs="+",
        help="One or more structure files (PDB or mmCIF)",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="Model index to use (1-based, default: 1)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--min-aligned",
        type=int,
        default=10,
        help="Minimum aligned residues to consider chains 'like' (default: 10)",
    )
    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=10.0,
        help="Maximum RMSD (Å) to consider chains 'like' (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        help="Output filename for CSV format (default: <stem>_all2all.csv)",
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
        help="Minimum B-factor threshold (default: -inf)",
    )
    parser.add_argument(
        "--max-bfac",
        type=float,
        default=np.inf,
        help="Maximum B-factor threshold (default: inf)",
    )
    parser.add_argument(
        "--min-occ",
        type=float,
        default=-np.inf,
        help="Minimum occupancy threshold (default: -inf)",
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
        help="Outlier rejection cutoff (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.filter and (args.min_bfac > args.max_bfac):
        parser.error("--min-bfac cannot be greater than --max-bfac")

    if args.model < 1:
        parser.error("--model must be >= 1")

    if args.format == "csv" and args.output is not None and not args.output.strip():
        parser.error("--output cannot be empty")

    if args.format == "table" and args.output is not None:
        parser.error("--output is only valid with --format csv")

    if args.max_rmsd <= 0:
        parser.error("--max-rmsd must be > 0")

    return args


def main() -> int:
    """Entry point for all-to-all chain alignment CLI."""
    args = _parse_args()
    _configure_logging(args.verbose)

    logger.info("=== All-to-All Chain Alignment ===\n")

    structure_paths = [Path(path) for path in args.structure]
    multiple_structures = len(structure_paths) > 1

    # Load structures and extract chains
    chains = []
    try:
        for structure_path in structure_paths:
            st = load_structure(structure_path)
            logger.info("Loaded structure: %s", structure_path.name)

            num_models = len(st)
            if num_models == 0:
                raise ValueError(f"Structure {structure_path.name} contains no models")
            if num_models > 1:
                logger.warning(
                    "Structure %s contains %d models; using model %d.",
                    structure_path.name,
                    num_models,
                    args.model,
                )

            if args.model > num_models:
                raise ValueError(f"Requested model {args.model} but {structure_path.name} has {num_models} model(s)")

            # Extract all protein chains
            structure_chains = get_all_protein_chains(st[args.model - 1])
            if not structure_chains:
                raise ValueError(f"No protein chains found in {structure_path.name}")

            if multiple_structures:
                for chain in structure_chains:
                    chain.chain_id = f"{structure_path.stem}_{chain.chain_id}"

            chains.extend(structure_chains)

        chains.sort(key=lambda chain: chain.chain_id)
        logger.info("Found %d protein chain(s) total.\n", len(chains))

    except Exception as e:
        logger.error("Error loading structure(s): %s", e)
        return 1

    # Perform all-to-all alignment
    results: list[tuple[str, str, int | None, float | None, str]] = []

    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            chain_i = chains[i]
            chain_j = chains[j]

            # Check if sequences align with sufficient overlap
            pairs = align_sequences(chain_i.sequence, chain_j.sequence)
            num_pairs = sum(1 for p1, p2 in pairs if p1 is not None and p2 is not None)

            if num_pairs < args.min_aligned:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Skipping %s vs %s: only %d aligned residues (< %d)",
                        chain_i.chain_id,
                        chain_j.chain_id,
                        num_pairs,
                        args.min_aligned,
                    )
                results.append((chain_i.chain_id, chain_j.chain_id, None, None, "min_aligned"))
                continue

            # Perform alignment
            try:
                _, _, rmsd, num_aligned = align_two_chains(
                    chain_i,
                    chain_j,
                    refine=args.refine,
                    max_cycles=args.cycles,
                    cutoff_factor=args.cutoff,
                    filter=args.filter,
                    min_bfactor=args.min_bfac,
                    max_bfactor=args.max_bfac,
                    min_occ=args.min_occ,
                )
                if num_aligned < args.min_aligned:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Skipping %s vs %s: only %d aligned residues (< %d) after filtering/refine",
                            chain_i.chain_id,
                            chain_j.chain_id,
                            num_aligned,
                            args.min_aligned,
                        )
                    results.append((chain_i.chain_id, chain_j.chain_id, None, None, "min_aligned"))
                    continue
                if rmsd > args.max_rmsd:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Skipping %s vs %s: RMSD %.3f Å (> %.3f)",
                            chain_i.chain_id,
                            chain_j.chain_id,
                            rmsd,
                            args.max_rmsd,
                        )
                    results.append((chain_i.chain_id, chain_j.chain_id, None, None, "max_rmsd"))
                    continue

                results.append((chain_i.chain_id, chain_j.chain_id, num_aligned, rmsd, "ok"))

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Aligned %s vs %s: %d residues, RMSD %.3f Å",
                        chain_i.chain_id,
                        chain_j.chain_id,
                        num_aligned,
                        rmsd,
                    )

            except Exception as e:
                logger.warning("Failed to align %s vs %s: %s", chain_i.chain_id, chain_j.chain_id, e)
                results.append((chain_i.chain_id, chain_j.chain_id, None, None, "error"))
                continue

    # Output results
    if not results:
        logger.info("No chain pairs available for alignment.\n")
        return 0

    results.sort(key=lambda row: (row[0], row[1]))

    if args.format == "csv":
        if args.output:
            output_path = Path(args.output)
        elif len(structure_paths) == 1:
            output_path = structure_paths[0].with_suffix("").with_name(f"{structure_paths[0].stem}_all2all.csv")
        else:
            output_path = Path("all2all.csv")
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("chain_1,chain_2,num_aligned,rmsd,status\n")
            for chain_id_i, chain_id_j, _num_aligned, _rmsd, status in results:
                if _num_aligned is None or _rmsd is None:
                    handle.write(f"{chain_id_i},{chain_id_j},,,{status}\n")
                else:
                    handle.write(f"{chain_id_i},{chain_id_j},{_num_aligned},{_rmsd:.3f},{status}\n")
        logger.info("Wrote CSV output: %s\n", output_path)
    else:
        logger.info("")
        # Table format
        rows: list[tuple[str, str, str, str, str]] = []
        for chain_id_i, chain_id_j, _num_aligned, _rmsd, _status in results:
            aligned_str = "---" if _num_aligned is None else str(_num_aligned)
            rmsd_str = "---" if _rmsd is None else f"{_rmsd:.3f}"
            status_str = "" if _status == "ok" else _status
            rows.append((chain_id_i, chain_id_j, aligned_str, rmsd_str, status_str))

        chain_1_width = max(len("Chain 1"), *(len(row[0]) for row in rows))
        chain_2_width = max(len("Chain 2"), *(len(row[1]) for row in rows))
        aligned_width = max(len("Aligned"), *(len(row[2]) for row in rows))
        rmsd_width = max(len("RMSD (Å)"), *(len(row[3]) for row in rows))
        status_width = max(len("Status"), *(len(row[4]) for row in rows))

        header = (
            f"{'Chain 1':<{chain_1_width}} {'Chain 2':<{chain_2_width}} "
            f"{'Aligned':<{aligned_width}} {'RMSD (Å)':<{rmsd_width}} {'Status':<{status_width}}"
        )
        print(header)
        print("-" * len(header))
        for chain_id_i, chain_id_j, aligned_str, rmsd_str, status_str in rows:
            print(
                f"{chain_id_i:<{chain_1_width}} {chain_id_j:<{chain_2_width}} "
                f"{aligned_str:<{aligned_width}} {rmsd_str:<{rmsd_width}} {status_str:<{status_width}}"
            )

    total_aligned = sum(
        1 for _, _, _num_aligned, _rmsd, _status in results if _num_aligned is not None and _rmsd is not None
    )
    logger.info("\nTotal pairs aligned: %d (of %d total)\n", total_aligned, len(results))
    return 0
