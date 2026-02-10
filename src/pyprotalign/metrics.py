"""Additional structural similarity scores (TM-score and GDT variants)."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def _pair_distances(fixed: NDArray[np.floating], mobile: NDArray[np.floating]) -> NDArray[np.floating]:
    fixed = np.asarray(fixed, dtype=float)
    mobile = np.asarray(mobile, dtype=float)
    if fixed.shape != mobile.shape:
        raise ValueError(f"Shape mismatch: fixed {fixed.shape} vs mobile {mobile.shape}")
    if fixed.ndim != 2 or fixed.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) arrays, got shape {fixed.shape}")
    if fixed.shape[0] == 0:
        raise ValueError("Need at least 1 aligned pair to compute score")
    return cast(NDArray[np.floating], np.sqrt(np.sum((fixed - mobile) ** 2, axis=1)))


def _tm_d0(l_target: int) -> float:
    if l_target <= 0:
        raise ValueError(f"l_target must be > 0, got {l_target}")
    if l_target <= 21:
        return 0.5
    return float(max(0.5, 1.24 * (l_target - 15) ** (1.0 / 3.0) - 1.8))


def tm_score(
    fixed: NDArray[np.floating],
    mobile: NDArray[np.floating],
    l_target: int | None = None,
) -> float:
    """Compute TM-score on aligned coordinates.

    Args:
        fixed: Fixed coordinates, shape (N, 3)
        mobile: Mobile coordinates already in fixed frame, shape (N, 3)
        l_target: Target length for TM normalization. Defaults to N.
    """
    distances = _pair_distances(fixed, mobile)
    n_aligned = distances.shape[0]
    norm_len = n_aligned if l_target is None else l_target
    d0 = _tm_d0(norm_len)
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / norm_len)


def gdt(
    fixed: NDArray[np.floating],
    mobile: NDArray[np.floating],
    cutoffs: tuple[float, ...],
    l_target: int | None = None,
) -> float:
    """Compute generic GDT score using provided distance cutoffs.

    Args:
        fixed: Fixed coordinates, shape (N, 3)
        mobile: Mobile coordinates already in fixed frame, shape (N, 3)
        cutoffs: Distance cutoffs in Å
        l_target: Target/reference length for normalization. Defaults to N.
    """
    if len(cutoffs) == 0:
        raise ValueError("cutoffs cannot be empty")
    distances = _pair_distances(fixed, mobile)
    n_aligned = distances.shape[0]
    norm_len = n_aligned if l_target is None else l_target
    if norm_len <= 0:
        raise ValueError(f"l_target must be > 0, got {norm_len}")
    fractions = [float(np.sum(distances <= cutoff) / norm_len) for cutoff in cutoffs]
    return float(np.mean(fractions))


def gdt_ts(fixed: NDArray[np.floating], mobile: NDArray[np.floating], l_target: int | None = None) -> float:
    """Compute GDT-TS score using cutoffs 1, 2, 4, and 8 Å."""
    return gdt(fixed, mobile, cutoffs=(1.0, 2.0, 4.0, 8.0), l_target=l_target)


def gdt_ha(fixed: NDArray[np.floating], mobile: NDArray[np.floating], l_target: int | None = None) -> float:
    """Compute GDT-HA score using cutoffs 0.5, 1, 2, and 4 Å."""
    return gdt(fixed, mobile, cutoffs=(0.5, 1.0, 2.0, 4.0), l_target=l_target)
