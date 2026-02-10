"""All-to-all chain alignment CLI tool."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from . import __version__
from .alignment import align_two_chains, collect_aligned_pair_coords, collect_quality_mask_for_pair
from .chain import ProteinChain
from .gemmi_utils import align_sequences, get_all_protein_chains, load_structure
from .metrics import gdt_ha, gdt_ts, tm_score

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
    parser.add_argument(
        "--score-scope",
        choices=["mapped", "filtered", "refined"],
        default="mapped",
        help="Residue scope for score calculation (default: mapped)",
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
    if args.score_scope == "refined" and not args.refine:
        parser.error("--score-scope refined requires --refine")

    return args


def _compute_pair_scores(
    chain_i: ProteinChain,
    chain_j: ProteinChain,
    rotation: np.ndarray,
    translation: np.ndarray,
    rmsd: float,
    cutoff_factor: float,
    score_scope: str,
    min_bfac: float,
    max_bfac: float,
    min_occ: float,
    refine_enabled: bool,
) -> tuple[float, float, float]:
    fixed_coords, mobile_coords, _, _ = collect_aligned_pair_coords(chain_i, chain_j)
    mobile_transformed = mobile_coords @ rotation.T + translation

    if score_scope == "filtered":
        quality_mask = collect_quality_mask_for_pair(
            chain_i,
            chain_j,
            min_bfactor=min_bfac,
            max_bfactor=max_bfac,
            min_occ=min_occ,
        )
        if np.sum(quality_mask) == 0:
            raise ValueError("No aligned CA pairs left after quality filtering for score calculation.")
        fixed_used = fixed_coords[quality_mask]
        mobile_used = mobile_transformed[quality_mask]
    elif score_scope == "refined":
        if not refine_enabled:
            raise ValueError("Internal error: refined score scope requires refinement.")
        distances = np.sqrt(np.sum((fixed_coords - mobile_transformed) ** 2, axis=1))
        mask = distances <= cutoff_factor * rmsd
        if np.sum(mask) == 0:
            raise ValueError("No aligned CA pairs left after refined score masking.")
        fixed_used = fixed_coords[mask]
        mobile_used = mobile_transformed[mask]
    else:
        fixed_used = fixed_coords
        mobile_used = mobile_transformed

    l_target = len(chain_i.sequence)
    tm = tm_score(fixed_used, mobile_used, l_target=l_target)
    gdt_ts_value = gdt_ts(fixed_used, mobile_used, l_target=l_target)
    gdt_ha_value = gdt_ha(fixed_used, mobile_used, l_target=l_target)
    return tm, gdt_ts_value, gdt_ha_value


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
    results: list[tuple[str, str, int | None, float | None, float | None, float | None, float | None, str]] = []

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
                results.append((chain_i.chain_id, chain_j.chain_id, None, None, None, None, None, "min_aligned"))
                continue

            # Perform alignment
            try:
                rotation, translation, rmsd, num_aligned = align_two_chains(
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
                    results.append((chain_i.chain_id, chain_j.chain_id, None, None, None, None, None, "min_aligned"))
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
                    results.append((chain_i.chain_id, chain_j.chain_id, None, None, None, None, None, "max_rmsd"))
                    continue

                tm, gdt_ts_value, gdt_ha_value = _compute_pair_scores(
                    chain_i,
                    chain_j,
                    rotation,
                    translation,
                    rmsd,
                    cutoff_factor=args.cutoff,
                    score_scope=args.score_scope,
                    min_bfac=args.min_bfac,
                    max_bfac=args.max_bfac,
                    min_occ=args.min_occ,
                    refine_enabled=args.refine,
                )

                results.append(
                    (
                        chain_i.chain_id,
                        chain_j.chain_id,
                        num_aligned,
                        rmsd,
                        tm,
                        gdt_ts_value,
                        gdt_ha_value,
                        "ok",
                    )
                )

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Aligned %s vs %s: %d residues, RMSD %.3f Å, TM %.4f, GDT-TS %.4f, GDT-HA %.4f",
                        chain_i.chain_id,
                        chain_j.chain_id,
                        num_aligned,
                        rmsd,
                        tm,
                        gdt_ts_value,
                        gdt_ha_value,
                    )

            except Exception as e:
                logger.warning("Failed to align %s vs %s: %s", chain_i.chain_id, chain_j.chain_id, e)
                results.append((chain_i.chain_id, chain_j.chain_id, None, None, None, None, None, "error"))
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
            handle.write("chain_1,chain_2,num_aligned,rmsd,tm,gdt_ts,gdt_ha,status\n")
            for chain_id_i, chain_id_j, _num_aligned, _rmsd, _tm, _gdt_ts, _gdt_ha, status in results:
                if _num_aligned is None or _rmsd is None or _tm is None or _gdt_ts is None or _gdt_ha is None:
                    handle.write(f"{chain_id_i},{chain_id_j},,,,,,{status}\n")
                else:
                    handle.write(
                        f"{chain_id_i},{chain_id_j},{_num_aligned},{_rmsd:.3f},{_tm:.4f},{_gdt_ts:.4f},{_gdt_ha:.4f},{status}\n"
                    )
        logger.info("Wrote CSV output: %s\n", output_path)
    else:
        logger.info("")
        # Table format
        rows: list[tuple[str, str, str, str, str, str, str, str]] = []
        for chain_id_i, chain_id_j, _num_aligned, _rmsd, _tm, _gdt_ts, _gdt_ha, _status in results:
            aligned_str = "---" if _num_aligned is None else str(_num_aligned)
            rmsd_str = "---" if _rmsd is None else f"{_rmsd:.3f}"
            tm_str = "---" if _tm is None else f"{_tm:.4f}"
            gdt_ts_str = "---" if _gdt_ts is None else f"{_gdt_ts:.4f}"
            gdt_ha_str = "---" if _gdt_ha is None else f"{_gdt_ha:.4f}"
            status_str = "" if _status == "ok" else _status
            rows.append((chain_id_i, chain_id_j, aligned_str, rmsd_str, tm_str, gdt_ts_str, gdt_ha_str, status_str))

        chain_1_width = max(len("Chain 1"), *(len(row[0]) for row in rows))
        chain_2_width = max(len("Chain 2"), *(len(row[1]) for row in rows))
        aligned_width = max(len("Aligned"), *(len(row[2]) for row in rows))
        rmsd_width = max(len("RMSD (Å)"), *(len(row[3]) for row in rows))
        tm_width = max(len("TM"), *(len(row[4]) for row in rows))
        gdt_ts_width = max(len("GDT-TS"), *(len(row[5]) for row in rows))
        gdt_ha_width = max(len("GDT-HA"), *(len(row[6]) for row in rows))
        status_width = max(len("Status"), *(len(row[7]) for row in rows))

        header = (
            f"{'Chain 1':<{chain_1_width}} {'Chain 2':<{chain_2_width}} "
            f"{'Aligned':<{aligned_width}} {'RMSD (Å)':<{rmsd_width}} {'TM':<{tm_width}} "
            f"{'GDT-TS':<{gdt_ts_width}} {'GDT-HA':<{gdt_ha_width}} {'Status':<{status_width}}"
        )
        print(header)
        print("-" * len(header))
        for chain_id_i, chain_id_j, aligned_str, rmsd_str, tm_str, gdt_ts_str, gdt_ha_str, status_str in rows:
            print(
                f"{chain_id_i:<{chain_1_width}} {chain_id_j:<{chain_2_width}} "
                f"{aligned_str:<{aligned_width}} {rmsd_str:<{rmsd_width}} {tm_str:<{tm_width}} "
                f"{gdt_ts_str:<{gdt_ts_width}} {gdt_ha_str:<{gdt_ha_width}} {status_str:<{status_width}}"
            )

    total_aligned = sum(
        1
        for _, _, _num_aligned, _rmsd, _tm, _gdt_ts, _gdt_ha, _status in results
        if all(value is not None for value in (_num_aligned, _rmsd, _tm, _gdt_ts, _gdt_ha))
    )
    logger.info("\nTotal pairs aligned: %d (of %d total)\n", total_aligned, len(results))
    return 0
