"""Sequence-independent structural alignment by iterative TM-score optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .chain import ProteinChain
from .gemmi_utils import align_sequences
from .kabsch import superpose
from .metrics import tm_score

logger = logging.getLogger(__name__)


@dataclass
class _TmState:
    rotation: NDArray[np.floating]
    translation: NDArray[np.floating]
    mapping: list[tuple[int, int]]
    score: float


def _compute_d0(l_target: int) -> float:
    if l_target <= 0:
        raise ValueError(f"l_target must be > 0, got {l_target}")
    if l_target <= 21:
        return 0.5
    return float(max(0.5, 1.24 * (l_target - 15) ** (1.0 / 3.0) - 1.8))


def _extract_valid_ca(chain: ProteinChain) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    valid_mask = np.all(np.isfinite(chain.coords), axis=1)
    if np.sum(valid_mask) < 3:
        raise ValueError(f"Chain '{chain.chain_id}' must contain at least 3 finite CA coordinates")
    coords = chain.coords[valid_mask]
    residue_indices = np.where(valid_mask)[0]
    return coords, residue_indices


def _seed_start_positions(length: int, window: int, max_positions: int = 5) -> list[int]:
    if window > length:
        return []
    max_start = length - window
    if max_start == 0:
        return [0]
    n_positions = min(max_positions, max_start + 1)
    positions = np.linspace(0, max_start, num=n_positions, dtype=int)
    return sorted({int(pos) for pos in positions})


def _build_multi_seed_mappings(n_fixed: int, n_mobile: int) -> list[list[tuple[int, int]]]:
    """Generate multiple diagonal fragment seeds across chain lengths."""
    seed_mappings: list[list[tuple[int, int]]] = []
    min_len = min(n_fixed, n_mobile)
    fragment_lengths = sorted(
        {
            min_len,
            min(100, min_len),
            min(50, min_len),
        },
        reverse=True,
    )
    for window in fragment_lengths:
        if window < 3:
            continue
        starts_fixed = _seed_start_positions(n_fixed, window)
        starts_mobile = _seed_start_positions(n_mobile, window)
        for start_fixed in starts_fixed:
            for start_mobile in starts_mobile:
                seed_mappings.append([(start_fixed + offset, start_mobile + offset) for offset in range(window)])
    return seed_mappings


def _assign_secondary_states(valid_coords: NDArray[np.floating]) -> str:
    """Assign rough 3-state secondary structure from CA geometry for seed initialization."""
    n = valid_coords.shape[0]
    if n == 0:
        return ""

    # CA(i)-CA(i+3) captures local compactness:
    # ~5-6.5A in helices and typically >8A in extended strands.
    dist_i_i3 = np.linalg.norm(valid_coords[:-3] - valid_coords[3:], axis=1)
    helix_mask = (dist_i_i3 >= 4.6) & (dist_i_i3 <= 6.5)
    strand_mask = (dist_i_i3 >= 8.2) & (dist_i_i3 <= 12.0)
    kernel = np.ones(4, dtype=int)
    helix_votes = np.convolve(helix_mask.astype(int), kernel, mode="full")
    strand_votes = np.convolve(strand_mask.astype(int), kernel, mode="full")

    # Keep only stable runs (>=3 residues) for H/E.
    # This is a crude heuristic to reduce noise in the initial seed assignment.
    helices = _remove_short_runs((helix_votes > strand_votes) & (helix_votes > 0), 3)
    strands = _remove_short_runs((strand_votes > helix_votes) & (strand_votes > 0), 3)

    states = np.full(valid_coords.shape[0], "C", dtype="<U1")
    states[helices] = "H"
    states[strands] = "E"

    return "".join(states)


def _remove_short_runs(arr: NDArray[np.bool_], min_len: int) -> NDArray[np.bool_]:
    """Remove runs of True values shorter than min_len from a boolean array using erosion/dilation."""
    if min_len <= 1:
        return arr

    kernel = np.ones(min_len, dtype=int)

    # 1. Erosion: Only positions that are part of a run of at least min_len True values will remain True.
    eroded = np.convolve(arr.astype(int), kernel, mode="valid") == min_len

    # 2. Dilation: Expand the remaining True values back to their original width to restore the full runs.
    dilated = np.convolve(eroded.astype(int), kernel, mode="full") > 0

    return dilated


def _ss_pair_score(fixed_ss: str, mobile_ss: str) -> float:
    """Score a secondary-structure state pair for DP alignment."""
    if fixed_ss == mobile_ss:
        return 2.0 if fixed_ss in {"H", "E"} else 0.5
    if fixed_ss == "C" or mobile_ss == "C":
        return -0.5
    return -1.0


def _traceback_secondary_pairs(
    fixed_ss: str,
    mobile_ss: str,
    dp: NDArray[np.floating],
    ptr_i: NDArray[np.int_],
    ptr_j: NDArray[np.int_],
    start_i: int,
    start_j: int,
    stop_at_zero: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Trace back one SS alignment path and collect pair-quality buckets."""
    diag_pairs: list[tuple[int, int]] = []
    exact_pairs: list[tuple[int, int]] = []
    structured_pairs: list[tuple[int, int]] = []

    i = start_i
    j = start_j
    while i > 0 and j > 0:
        if stop_at_zero and dp[i, j] <= 0.0:
            break
        ni = int(ptr_i[i, j])
        nj = int(ptr_j[i, j])
        if ni == i and nj == j:
            break

        if ni == i - 1 and nj == j - 1:
            fixed_state = fixed_ss[i - 1]
            mobile_state = mobile_ss[j - 1]
            diag_pairs.append((i - 1, j - 1))
            if fixed_state == mobile_state:
                exact_pairs.append((i - 1, j - 1))
                if fixed_state in {"H", "E"}:
                    structured_pairs.append((i - 1, j - 1))
        i, j = ni, nj

    diag_pairs.reverse()
    exact_pairs.reverse()
    structured_pairs.reverse()

    return diag_pairs, exact_pairs, structured_pairs


def _select_secondary_pairs(
    diag_pairs: list[tuple[int, int]],
    exact_pairs: list[tuple[int, int]],
    structured_pairs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Select best pair set with graceful degradation if high-quality sets are short."""
    if len(structured_pairs) >= 3:
        return structured_pairs
    if len(exact_pairs) >= 3:
        return exact_pairs
    return diag_pairs


def _align_secondary_states_local(
    fixed_ss: str, mobile_ss: str, gap_penalty: float = -1.0
) -> tuple[float, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Align SS strings using local DP, returning the best score and seed mappings."""
    n_fixed = len(fixed_ss)
    n_mobile = len(mobile_ss)
    if n_fixed == 0 or n_mobile == 0:
        return -np.inf, [], [], []

    dp = np.zeros((n_fixed + 1, n_mobile + 1), dtype=float)
    ptr_i = np.zeros((n_fixed + 1, n_mobile + 1), dtype=int)
    ptr_j = np.zeros((n_fixed + 1, n_mobile + 1), dtype=int)
    best_score = 0.0
    best_i = 0
    best_j = 0

    for i in range(1, n_fixed + 1):
        for j in range(1, n_mobile + 1):
            diag = dp[i - 1, j - 1] + _ss_pair_score(fixed_ss[i - 1], mobile_ss[j - 1])
            up = dp[i - 1, j] + gap_penalty
            left = dp[i, j - 1] + gap_penalty
            best = max(0.0, diag, up, left)
            dp[i, j] = best
            if best == 0.0:
                ptr_i[i, j] = i
                ptr_j[i, j] = j
            elif best == diag:
                ptr_i[i, j] = i - 1
                ptr_j[i, j] = j - 1
            elif best == up:
                ptr_i[i, j] = i - 1
                ptr_j[i, j] = j
            else:
                ptr_i[i, j] = i
                ptr_j[i, j] = j - 1

            if best > best_score:
                best_score = best
                best_i = i
                best_j = j

    return best_score, *_traceback_secondary_pairs(
        fixed_ss,
        mobile_ss,
        dp,
        ptr_i,
        ptr_j,
        best_i,
        best_j,
        stop_at_zero=True,
    )


def _align_secondary_states_semiglobal(
    fixed_ss: str, mobile_ss: str, gap_penalty: float = -1.0
) -> tuple[float, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Align SS strings using semi-global DP, returning the best score and seed mappings."""
    n_fixed = len(fixed_ss)
    n_mobile = len(mobile_ss)
    if n_fixed == 0 or n_mobile == 0:
        return -np.inf, [], [], []

    dp = np.zeros((n_fixed + 1, n_mobile + 1), dtype=float)
    ptr_i = np.zeros((n_fixed + 1, n_mobile + 1), dtype=int)
    ptr_j = np.zeros((n_fixed + 1, n_mobile + 1), dtype=int)
    for i in range(1, n_fixed + 1):
        ptr_i[i, 0] = i - 1
        ptr_j[i, 0] = 0
    for j in range(1, n_mobile + 1):
        ptr_i[0, j] = 0
        ptr_j[0, j] = j - 1

    for i in range(1, n_fixed + 1):
        for j in range(1, n_mobile + 1):
            diag = dp[i - 1, j - 1] + _ss_pair_score(fixed_ss[i - 1], mobile_ss[j - 1])
            up = dp[i - 1, j] + gap_penalty
            left = dp[i, j - 1] + gap_penalty
            best = max(diag, up, left)
            dp[i, j] = best
            if best == diag:
                ptr_i[i, j] = i - 1
                ptr_j[i, j] = j - 1
            elif best == up:
                ptr_i[i, j] = i - 1
                ptr_j[i, j] = j
            else:
                ptr_i[i, j] = i
                ptr_j[i, j] = j - 1

    last_line_max_score = float(np.max(dp[n_fixed, :]))
    last_col_max_score = float(np.max(dp[:, n_mobile]))
    if last_line_max_score >= last_col_max_score:
        best_i = n_fixed
        best_j = int(np.argmax(dp[n_fixed]))
    else:
        best_j = n_mobile
        best_i = int(np.argmax(dp[:, n_mobile]))
    best_score = max(last_line_max_score, last_col_max_score)

    return best_score, *_traceback_secondary_pairs(
        fixed_ss,
        mobile_ss,
        dp,
        ptr_i,
        ptr_j,
        best_i,
        best_j,
        stop_at_zero=False,
    )


def _align_secondary_states(
    fixed_ss: str, mobile_ss: str, gap_penalty: float = -1.0
) -> tuple[str, list[tuple[int, int]]]:
    """Align SS strings using local and semi-global DP, returning the better seed mapping."""
    n_fixed = len(fixed_ss)
    n_mobile = len(mobile_ss)
    if n_fixed == 0 or n_mobile == 0:
        return "none", []

    local_best_score, local_diag, local_exact, local_structured = _align_secondary_states_local(
        fixed_ss, mobile_ss, gap_penalty=gap_penalty
    )

    semi_best_score, semi_diag, semi_exact, semi_structured = _align_secondary_states_semiglobal(
        fixed_ss, mobile_ss, gap_penalty=gap_penalty
    )

    local_selected = _select_secondary_pairs(local_diag, local_exact, local_structured)
    semi_selected = _select_secondary_pairs(semi_diag, semi_exact, semi_structured)

    local_rank = (local_best_score, len(local_structured), len(local_exact), len(local_diag))
    semi_rank = (semi_best_score, len(semi_structured), len(semi_exact), len(semi_diag))

    return ("local", local_selected) if local_rank >= semi_rank else ("semi-global", semi_selected)


def _build_secondary_seed_mapping(
    fixed_coords: NDArray[np.floating], mobile_coords: NDArray[np.floating]
) -> list[tuple[int, int]]:
    """Generate a single SS-based seed mapping from CA coordinates."""
    fixed_ss = _assign_secondary_states(fixed_coords)
    mobile_ss = _assign_secondary_states(mobile_coords)
    align_type, mapping = _align_secondary_states(fixed_ss, mobile_ss)
    structured_matches = sum(1 for i, j in mapping if fixed_ss[i] == mobile_ss[j] and fixed_ss[i] in {"H", "E"})

    if logger.isEnabledFor(logging.DEBUG):
        fixed_h = fixed_ss.count("H")
        fixed_e = fixed_ss.count("E")
        mobile_h = mobile_ss.count("H")
        mobile_e = mobile_ss.count("E")
        logger.debug(
            "TM-align secondary seeds: '%s', fixed(H=%d,E=%d), mobile(H=%d,E=%d), mapped=%d",
            align_type,
            fixed_h,
            fixed_e,
            mobile_h,
            mobile_e,
            len(mapping),
        )

    # If the mapping has almost no matched H/E signal, SS-based seeding is underconstrained.
    # Returning an empty mapping triggers robust multi-seed fallback.
    if structured_matches < 3:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "TM-align secondary seeds underconstrained (matched H/E=%d); using fallback seeds",
                structured_matches,
            )
        return []
    return mapping


def _build_sequence_seed_mapping(
    fixed_chain: ProteinChain,
    mobile_chain: ProteinChain,
    fixed_res_idx: NDArray[np.int_],
    mobile_res_idx: NDArray[np.int_],
) -> list[tuple[int, int]]:
    """Generate one seed mapping from sequence alignment projected to valid-CA indices."""
    fixed_seq_to_valid = np.full(len(fixed_chain.sequence), -1, dtype=int)
    mobile_seq_to_valid = np.full(len(mobile_chain.sequence), -1, dtype=int)
    fixed_seq_to_valid[fixed_res_idx] = np.arange(fixed_res_idx.shape[0], dtype=int)
    mobile_seq_to_valid[mobile_res_idx] = np.arange(mobile_res_idx.shape[0], dtype=int)

    mapping: list[tuple[int, int]] = []
    for fixed_seq_i, mobile_seq_i in align_sequences(fixed_chain.sequence, mobile_chain.sequence):
        if fixed_seq_i is None or mobile_seq_i is None:
            continue
        fixed_valid_i = fixed_seq_to_valid[fixed_seq_i]
        mobile_valid_i = mobile_seq_to_valid[mobile_seq_i]
        if fixed_valid_i >= 0 and mobile_valid_i >= 0:
            mapping.append((int(fixed_valid_i), int(mobile_valid_i)))
    return mapping


def _build_seed_mappings(
    fixed_chain: ProteinChain,
    mobile_chain: ProteinChain,
    fixed_res_idx: NDArray[np.int_],
    mobile_res_idx: NDArray[np.int_],
    seeds: Literal["single", "multi", "sequence", "secondary"],
) -> list[list[tuple[int, int]]]:
    """Build seed mappings for requested mode."""
    n_fixed = fixed_res_idx.shape[0]
    n_mobile = mobile_res_idx.shape[0]
    min_len = min(n_fixed, n_mobile)
    if min_len < 3:
        return []

    if seeds == "single":
        window = min_len
        return [[(k, k) for k in range(window)]]
    if seeds == "sequence":
        mapping = _build_sequence_seed_mapping(fixed_chain, mobile_chain, fixed_res_idx, mobile_res_idx)
        return [mapping] if len(mapping) >= 3 else []
    if seeds == "secondary":
        mapping = _build_secondary_seed_mapping(fixed_chain.coords[fixed_res_idx], mobile_chain.coords[mobile_res_idx])
        if len(mapping) >= 3:
            return [mapping]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("TM-align secondary seeds too small (%d); falling back to multi seeds", len(mapping))
        return _build_multi_seed_mappings(n_fixed, n_mobile)
    if seeds != "multi":
        raise ValueError("seeds must be 'single', 'multi', 'sequence', or 'secondary'")

    seed_mappings = _build_multi_seed_mappings(n_fixed, n_mobile)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "TM-align seed generation: n_fixed=%d, n_mobile=%d, mode=%s, seeds=%d",
            n_fixed,
            n_mobile,
            seeds,
            len(seed_mappings),
        )

    return seed_mappings


def _compute_similarity_matrix(
    fixed_coords: NDArray[np.floating],
    mobile_transformed: NDArray[np.floating],
    d0: float,
) -> NDArray[np.floating]:
    deltas = fixed_coords[:, np.newaxis, :] - mobile_transformed[np.newaxis, :, :]
    dist_sq: NDArray[np.floating] = np.sum(deltas * deltas, axis=2)
    return 1.0 / (1.0 + dist_sq / d0**2)


def _dp_best_mapping(similarity: NDArray[np.floating], gap_penalty: float = 0.0) -> list[tuple[int, int]]:
    n_fixed, n_mobile = similarity.shape
    dp = np.zeros((n_fixed + 1, n_mobile + 1), dtype=float)
    ptr_i = np.zeros((n_fixed + 1, n_mobile + 1), dtype=int)
    ptr_j = np.zeros((n_fixed + 1, n_mobile + 1), dtype=int)

    best_score = 0.0
    best_i = 0
    best_j = 0

    for i in range(1, n_fixed + 1):
        for j in range(1, n_mobile + 1):
            diag = dp[i - 1, j - 1] + similarity[i - 1, j - 1]
            up = dp[i - 1, j] + gap_penalty
            left = dp[i, j - 1] + gap_penalty
            best = max(0.0, diag, up, left)
            dp[i, j] = best
            if best == 0.0:
                ptr_i[i, j] = 0
                ptr_j[i, j] = 0
            elif best == diag:
                ptr_i[i, j] = i - 1
                ptr_j[i, j] = j - 1
            elif best == up:
                ptr_i[i, j] = i - 1
                ptr_j[i, j] = j
            else:
                ptr_i[i, j] = i
                ptr_j[i, j] = j - 1

            if best > best_score:
                best_score = best
                best_i = i
                best_j = j

    mapping: list[tuple[int, int]] = []
    i = best_i
    j = best_j
    while i > 0 and j > 0 and dp[i, j] > 0.0:
        ni = ptr_i[i, j]
        nj = ptr_j[i, j]
        if ni == i - 1 and nj == j - 1:
            mapping.append((i - 1, j - 1))
        i, j = ni, nj
    mapping.reverse()
    return mapping


def _refit_from_mapping(
    fixed_coords: NDArray[np.floating],
    mobile_coords: NDArray[np.floating],
    mapping: list[tuple[int, int]],
    d0: float,
    rotation: NDArray[np.floating],
    translation: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if len(mapping) < 3:
        raise ValueError("Need at least 3 mapped pairs for superposition")

    fixed_idx = np.array([pair[0] for pair in mapping], dtype=int)
    mobile_idx = np.array([pair[1] for pair in mapping], dtype=int)
    fixed_mapped = fixed_coords[fixed_idx]
    mobile_mapped = mobile_coords[mobile_idx]

    mobile_prev = mobile_mapped @ rotation.T + translation
    dist_sq = np.sum((fixed_mapped - mobile_prev) ** 2, axis=1)
    weights = 1.0 / (1.0 + dist_sq / d0**2)
    return superpose(fixed_mapped, mobile_mapped, weights=weights)


def tm_align(
    fixed_chain: ProteinChain,
    mobile_chain: ProteinChain,
    max_iter: int = 20,
    seeds: Literal["single", "multi", "sequence", "secondary"] = "sequence",
    d0: float | None = None,
    early_stop_threshold: float = 0.95,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, int, tuple[tuple[int, int], ...]]:
    """Sequence-independent structural alignment optimizing TM-score.

    Returns:
        rotation: 3x3 rotation matrix
        translation: 3-element translation vector
        tm: Final TM-score (normalized by fixed sequence length)
        num_pairs: Number of mapped pairs used for final score
        mapping: Tuple of mapped residue index pairs (fixed_idx, mobile_idx)
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    fixed_coords, fixed_res_idx = _extract_valid_ca(fixed_chain)
    mobile_coords, mobile_res_idx = _extract_valid_ca(mobile_chain)

    l_target = len(fixed_chain.sequence)
    d0_value = _compute_d0(l_target) if d0 is None else d0
    if d0_value <= 0:
        raise ValueError(f"d0 must be > 0, got {d0_value}")

    if debug_enabled:
        logger.debug(
            "TM-align start: fixed=%s (%d residues, %d valid CA), mobile=%s (%d residues, %d valid CA), "
            "max_iter=%d, seeds=%s, d0=%.4f",
            fixed_chain.chain_id,
            len(fixed_chain.sequence),
            fixed_coords.shape[0],
            mobile_chain.chain_id,
            len(mobile_chain.sequence),
            mobile_coords.shape[0],
            max_iter,
            seeds,
            d0_value,
        )

    seed_mappings = _build_seed_mappings(fixed_chain, mobile_chain, fixed_res_idx, mobile_res_idx, seeds)
    if not seed_mappings:
        raise ValueError("Unable to generate valid seeds for TM-align")

    best: _TmState | None = None
    for seed_idx, seed_mapping in enumerate(seed_mappings, start=1):
        if len(seed_mapping) < 3:
            continue
        fixed_idx = np.array([pair[0] for pair in seed_mapping], dtype=int)
        mobile_idx = np.array([pair[1] for pair in seed_mapping], dtype=int)
        rotation, translation = superpose(fixed_coords[fixed_idx], mobile_coords[mobile_idx])

        if debug_enabled:
            logger.debug(
                "Seed %d/%d: initial_pairs=%d (fixed %d-%d, mobile %d-%d)",
                seed_idx,
                len(seed_mappings),
                len(seed_mapping),
                seed_mapping[0][0],
                seed_mapping[-1][0],
                seed_mapping[0][1],
                seed_mapping[-1][1],
            )

        mapping = seed_mapping
        prev_score = -1.0
        for iter_idx in range(1, max_iter + 1):
            mobile_transformed = mobile_coords @ rotation.T + translation
            similarity = _compute_similarity_matrix(fixed_coords, mobile_transformed, d0_value)
            mapping = _dp_best_mapping(similarity)
            if len(mapping) < 3:
                if debug_enabled:
                    logger.debug(
                        "  Seed %d iter %d: mapping too small (%d), stopping",
                        seed_idx,
                        iter_idx,
                        len(mapping),
                    )
                break

            rotation, translation = _refit_from_mapping(
                fixed_coords,
                mobile_coords,
                mapping,
                d0_value,
                rotation,
                translation,
            )

            fixed_map_idx = np.array([pair[0] for pair in mapping], dtype=int)
            mobile_map_idx = np.array([pair[1] for pair in mapping], dtype=int)
            mobile_fit = mobile_coords[mobile_map_idx] @ rotation.T + translation
            score = tm_score(fixed_coords[fixed_map_idx], mobile_fit, l_target=l_target)

            if debug_enabled:
                logger.debug("  Seed %d iter %d: mapped_pairs=%d, tm=%.6f", seed_idx, iter_idx, len(mapping), score)

            if abs(score - prev_score) < 1e-6:
                prev_score = score
                if debug_enabled:
                    logger.debug("  Seed %d iter %d: converged (delta < 1e-6)", seed_idx, iter_idx)
                break
            prev_score = score

        if len(mapping) < 3:
            continue

        fixed_map_idx = np.array([pair[0] for pair in mapping], dtype=int)
        mobile_map_idx = np.array([pair[1] for pair in mapping], dtype=int)
        mobile_fit = mobile_coords[mobile_map_idx] @ rotation.T + translation
        score = tm_score(fixed_coords[fixed_map_idx], mobile_fit, l_target=l_target)

        if debug_enabled:
            logger.debug("Seed %d final: mapped_pairs=%d, tm=%.6f", seed_idx, len(mapping), score)

        if best is None or score > best.score:
            best = _TmState(rotation=rotation, translation=translation, mapping=mapping, score=score)
            if debug_enabled:
                logger.debug("Seed %d: new best tm=%.6f", seed_idx, score)

        if best.score >= early_stop_threshold:
            if debug_enabled:
                logger.debug("Seed %d: early stopping (TM-score >= %.2f)", seed_idx, early_stop_threshold)
            break

    if best is None:
        raise ValueError("TM-align failed to find a valid mapping with at least 3 pairs")

    mapping_residue = tuple((int(fixed_res_idx[i]), int(mobile_res_idx[j])) for i, j in best.mapping)
    if debug_enabled:
        logger.debug("TM-align done: best_tm=%.6f, mapped_pairs=%d", best.score, len(mapping_residue))
    return best.rotation, best.translation, best.score, len(mapping_residue), mapping_residue
