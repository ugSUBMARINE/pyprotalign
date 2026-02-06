"""Kabsch algorithm for optimal structural superposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def superpose(
    fixed: NDArray[np.floating],
    mobile: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute optimal rotation and translation to superpose mobile onto fixed.

    Uses the Kabsch algorithm to find the optimal rigid-body transformation
    that minimizes the weighted RMSD between corresponding points.

    Args:
        fixed: Fixed coordinates, shape (N, 3)
        mobile: Mobile coordinates to transform, shape (N, 3)
        weights: Optional weights for each point, shape (N,). If None, uniform weights used.

    Returns:
        rotation: Optimal rotation matrix, shape (3, 3)
        translation: Translation vector to apply after rotation, shape (3,)

    Raises:
        ValueError: If arrays have incompatible shapes or invalid values

    Notes:
        To apply transformation: mobile_transformed = (mobile @ rotation.T) + translation
    """
    fixed = np.asarray(fixed, dtype=float)
    mobile = np.asarray(mobile, dtype=float)

    if fixed.shape != mobile.shape:
        raise ValueError(f"Shape mismatch: fixed {fixed.shape} vs mobile {mobile.shape}")
    if fixed.ndim != 2 or fixed.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) arrays, got shape {fixed.shape}")

    n_points = fixed.shape[0]
    if n_points < 3:
        raise ValueError(f"Need at least 3 points for superposition, got {n_points}")

    # Handle weights
    if weights is None:
        weights = np.ones(n_points, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_points,):
            raise ValueError(f"Weights shape {weights.shape} incompatible with coordinates ({n_points}, 3)")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        if np.sum(weights) <= 0:
            raise ValueError("Sum of weights must be positive")

    # Normalize weights
    weights = weights / np.sum(weights)

    # Compute weighted centroids
    fixed_centroid = np.sum(fixed * weights[:, np.newaxis], axis=0)
    mobile_centroid = np.sum(mobile * weights[:, np.newaxis], axis=0)

    # Center coordinates
    fixed_centered = fixed - fixed_centroid
    mobile_centered = mobile - mobile_centroid

    # Compute covariance matrix with weights
    covariance = (mobile_centered.T * weights) @ fixed_centered

    # SVD
    u, _, vh = np.linalg.svd(covariance)

    # Compute rotation matrix, handling reflection case
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1
        rotation = vh.T @ u.T

    # Translation: fixed_centroid - mobile_centroid @ rotation.T
    translation = fixed_centroid - mobile_centroid @ rotation.T

    return rotation, translation


def calculate_rmsd(
    fixed: NDArray[np.floating],
    mobile: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Calculate RMSD between corresponding points.

    Args:
        fixed: Fixed coordinates, shape (N, 3)
        mobile: Mobile coordinates, shape (N, 3)
        weights: Optional weights for each point, shape (N,). If None, uniform weights used.

    Returns:
        RMSD value

    Raises:
        ValueError: If arrays have incompatible shapes
    """
    fixed = np.asarray(fixed, dtype=float)
    mobile = np.asarray(mobile, dtype=float)

    if fixed.shape != mobile.shape:
        raise ValueError(f"Shape mismatch: fixed {fixed.shape} vs mobile {mobile.shape}")
    if fixed.ndim != 2 or fixed.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) arrays, got shape {fixed.shape}")

    n_points = fixed.shape[0]

    # Handle weights
    if weights is None:
        weights = np.ones(n_points, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_points,):
            raise ValueError(f"Weights shape {weights.shape} incompatible with coordinates ({n_points}, 3)")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        if np.sum(weights) <= 0:
            raise ValueError("Sum of weights must be positive")

    # Normalize weights
    weights = weights / np.sum(weights)

    # Calculate squared differences
    sq_diff = np.sum((fixed - mobile) ** 2, axis=1)

    # Weighted RMSD
    return float(np.sqrt(np.sum(weights * sq_diff)))
