"""Iterative refinement for structure superposition."""

import logging

import numpy as np
from numpy.typing import NDArray

from .kabsch import calculate_rmsd, superpose

logger = logging.getLogger(__name__)


def iterative_superpose(
    fixed: NDArray[np.floating],
    mobile: NDArray[np.floating],
    max_cycles: int = 5,
    cutoff_factor: float = 2.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_], float]:
    """Iteratively superpose structures, rejecting outliers each cycle.

    Similar to PyMOL's align command. Each cycle:
    1. Superpose using current atom pairs
    2. Calculate RMSD
    3. Reject pairs with distance > cutoff_factor * RMSD
    4. Repeat until RMSD stops improving or max_cycles reached

    Args:
        fixed: Fixed coordinates, shape (N, 3)
        mobile: Mobile coordinates, shape (N, 3)
        max_cycles: Maximum refinement cycles
        cutoff_factor: Reject pairs beyond cutoff_factor * RMSD
    Returns:
        rotation: Final rotation matrix, shape (3, 3)
        translation: Final translation vector, shape (3,)
        mask: Boolean mask of retained pairs, shape (N,)
        rmsd: Final RMSD

    Raises:
        ValueError: If too few pairs remain after filtering
    """
    fixed = np.asarray(fixed, dtype=float)
    mobile = np.asarray(mobile, dtype=float)

    n_points = fixed.shape[0]
    mask = np.ones(n_points, dtype=bool)
    prev_rmsd = float("inf")

    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    if debug_enabled:
        logger.debug("\n  Refinement cycles:")

    for cycle in range(max_cycles):
        # Superpose current subset
        rotation, translation = superpose(fixed[mask], mobile[mask])

        # Transform mobile coords
        mobile_transformed = mobile @ rotation.T + translation

        # Calculate RMSD for all pairs
        rmsd = calculate_rmsd(fixed[mask], mobile_transformed[mask])

        if debug_enabled:
            n_used = np.sum(mask)
            logger.debug("    Cycle %d: %d pairs, RMSD = %.3f Å", cycle + 1, n_used, rmsd)

        # Check convergence
        if rmsd >= prev_rmsd * 0.999:  # Allow 0.1% tolerance
            if debug_enabled:
                logger.debug("    Converged (RMSD stopped improving)")
            break

        prev_rmsd = rmsd

        # Calculate distances for all original pairs
        distances = np.sqrt(np.sum((fixed - mobile_transformed) ** 2, axis=1))

        # Update mask: reject outliers
        new_mask = distances <= cutoff_factor * rmsd
        n_rejected = np.sum(mask) - np.sum(new_mask)

        # Don't update mask if too many would be rejected
        if np.sum(new_mask) < 3:
            break

        if n_rejected == 0:
            # No more outliers, converged
            if debug_enabled:
                logger.debug("    Converged (no more outliers)")
            break

        mask = new_mask

    # Final superposition with retained pairs
    rotation, translation = superpose(fixed[mask], mobile[mask])
    mobile_transformed = mobile @ rotation.T + translation
    rmsd = calculate_rmsd(fixed[mask], mobile_transformed[mask])

    return rotation, translation, mask, rmsd
