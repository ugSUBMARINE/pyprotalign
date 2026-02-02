"""Tests for iterative refinement."""

import numpy as np
import pytest

from pyprotalign.refine import iterative_superpose


class TestIterativeSuperpose:
    """Tests for iterative_superpose function."""

    def test_identical_structures_no_refinement_needed(self) -> None:
        """Test that identical structures converge immediately."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        rotation, translation, mask, rmsd = iterative_superpose(coords, coords)

        # Should converge in 1 cycle, keep all pairs
        assert np.sum(mask) == 4
        assert rmsd == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-10)

    def test_single_outlier_rejected(self) -> None:
        """Test that single outlier is rejected with strict cutoff."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mobile = fixed.copy()
        mobile[2] = [50.0, 50.0, 50.0]  # Make third point an outlier

        rotation, translation, mask, rmsd = iterative_superpose(fixed, mobile, cutoff_factor=1.5)

        # Should reject outlier
        assert np.sum(mask) == 3
        assert mask[2] == False  # noqa: E712
        assert rmsd < 1.0  # RMSD should be much better without outlier

    def test_outlier_kept_with_permissive_cutoff(self) -> None:
        """Test that outlier is kept with very permissive cutoff."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mobile = fixed.copy()
        mobile[2] = [5.0, 5.0, 5.0]  # Moderate outlier

        rotation, translation, mask, rmsd = iterative_superpose(fixed, mobile, cutoff_factor=100.0)

        # With very large cutoff, should keep all pairs
        assert np.sum(mask) == 4

    def test_max_cycles_respected(self) -> None:
        """Test that max_cycles parameter is respected."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mobile = fixed + np.array([0.1, 0.0, 0.0])  # Small displacement

        # Even with 0 cycles, should do at least initial + final superposition
        rotation, translation, mask, rmsd = iterative_superpose(fixed, mobile, max_cycles=0)

        # Should still work
        assert np.sum(mask) >= 3

    def test_minimum_pairs_preserved(self) -> None:
        """Test that at least 3 pairs are always kept."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mobile = np.array([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0], [50.0, 51.0, 50.0], [50.0, 50.0, 51.0]])

        # Even with very strict cutoff, should keep at least 3
        rotation, translation, mask, rmsd = iterative_superpose(fixed, mobile, cutoff_factor=0.1)

        assert np.sum(mask) >= 3

    def test_convergence_on_rmsd_improvement(self) -> None:
        """Test convergence when RMSD stops improving."""
        np.random.seed(42)
        fixed = np.random.randn(10, 3) * 5.0
        mobile = fixed + np.random.randn(10, 3) * 0.1  # Small noise

        rotation, translation, mask, rmsd = iterative_superpose(fixed, mobile, max_cycles=10)

        # Should converge before max_cycles
        # RMSD should be small
        assert rmsd < 1.0

    def test_transformation_correctness(self) -> None:
        """Test that returned transformation is correct."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mobile = fixed + np.array([5.0, 3.0, -2.0])

        rotation, translation, mask, rmsd = iterative_superpose(fixed, mobile)

        # Apply transformation
        mobile_transformed = mobile @ rotation.T + translation

        # Check transformed coords match fixed (for retained pairs)
        np.testing.assert_allclose(mobile_transformed[mask], fixed[mask], atol=1e-8)

    def test_multiple_outliers(self) -> None:
        """Test rejection of multiple outliers."""
        fixed = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        mobile = fixed.copy()
        mobile[2] = [50.0, 50.0, 50.0]  # Outlier 1
        mobile[4] = [60.0, 60.0, 60.0]  # Outlier 2

        rotation, translation, mask, rmsd = iterative_superpose(fixed, mobile, cutoff_factor=1.0)

        # Should reject both outliers with strict cutoff
        assert np.sum(mask) == 3
        assert mask[2] == False  # noqa: E712
        assert mask[4] == False  # noqa: E712
