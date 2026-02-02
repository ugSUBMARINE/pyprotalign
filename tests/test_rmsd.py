"""Tests for RMSD calculation."""

import numpy as np
import pytest

from pyprotalign.kabsch import calculate_rmsd


class TestCalculateRMSD:
    """Tests for calculate_rmsd function."""

    def test_identical_structures(self) -> None:
        """Test RMSD is zero for identical structures."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rmsd = calculate_rmsd(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)

    def test_simple_displacement(self) -> None:
        """Test RMSD for uniform displacement."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mobile = fixed + np.array([1.0, 0.0, 0.0])  # Shift by 1 Angstrom in X

        rmsd = calculate_rmsd(fixed, mobile)
        assert rmsd == pytest.approx(1.0, abs=1e-10)

    def test_known_rmsd(self) -> None:
        """Test RMSD with known value."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mobile = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        # Distances: 0, 1, 1 -> RMSD = sqrt((0^2 + 1^2 + 1^2)/3) = sqrt(2/3)
        rmsd = calculate_rmsd(fixed, mobile)
        expected = np.sqrt(2.0 / 3.0)
        assert rmsd == pytest.approx(expected, abs=1e-10)

    def test_with_weights(self) -> None:
        """Test weighted RMSD calculation."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mobile = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # First point has distance 1, others 0
        # With equal weights: sqrt((1 + 0 + 0)/3) = sqrt(1/3)
        rmsd_uniform = calculate_rmsd(fixed, mobile)
        assert rmsd_uniform == pytest.approx(np.sqrt(1.0 / 3.0), abs=1e-10)

        # Weight first point highly
        weights = np.array([100.0, 1.0, 1.0])
        rmsd_weighted = calculate_rmsd(fixed, mobile, weights=weights)

        # Weighted: sqrt((100*1 + 1*0 + 1*0)/102) = sqrt(100/102)
        expected_weighted = np.sqrt(100.0 / 102.0)
        assert rmsd_weighted == pytest.approx(expected_weighted, abs=1e-10)

    def test_uniform_weights_equals_none(self) -> None:
        """Test uniform weights give same result as no weights."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

        rmsd1 = calculate_rmsd(fixed, mobile)
        rmsd2 = calculate_rmsd(fixed, mobile, weights=np.ones(3))

        assert rmsd1 == pytest.approx(rmsd2, abs=1e-10)

    def test_shape_mismatch_error(self) -> None:
        """Test error on shape mismatch."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mobile = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_rmsd(fixed, mobile)

    def test_wrong_dimensions_error(self) -> None:
        """Test error on wrong dimensions."""
        fixed = np.array([[1.0, 2.0], [3.0, 4.0]])
        mobile = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Expected \\(N, 3\\) arrays"):
            calculate_rmsd(fixed, mobile)

    def test_weights_shape_mismatch_error(self) -> None:
        """Test error when weights have wrong shape."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = fixed.copy()
        weights = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Weights shape .* incompatible"):
            calculate_rmsd(fixed, mobile, weights=weights)

    def test_negative_weights_error(self) -> None:
        """Test error on negative weights."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = fixed.copy()
        weights = np.array([1.0, -1.0, 1.0])

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            calculate_rmsd(fixed, mobile, weights=weights)

    def test_zero_sum_weights_error(self) -> None:
        """Test error when all weights sum to zero."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = fixed.copy()
        weights = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Sum of weights must be positive"):
            calculate_rmsd(fixed, mobile, weights=weights)
