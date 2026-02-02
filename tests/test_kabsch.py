"""Tests for Kabsch algorithm superposition."""

import numpy as np
import pytest

from pyprotalign.kabsch import superpose


class TestSuperpose:
    """Tests for superpose function."""

    def test_identity_transformation(self) -> None:
        """Test that identical structures give identity rotation and zero translation."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotation, translation = superpose(coords, coords)

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-10)

    def test_translation_only(self) -> None:
        """Test pure translation without rotation."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mobile = fixed + np.array([5.0, 3.0, -2.0])

        rotation, translation = superpose(fixed, mobile)

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(translation, np.array([-5.0, -3.0, 2.0]), atol=1e-10)

        # Verify transformation works
        transformed = mobile @ rotation.T + translation
        np.testing.assert_allclose(transformed, fixed, atol=1e-10)

    def test_rotation_90_degrees_z(self) -> None:
        """Test 90-degree rotation around Z-axis."""
        fixed = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Rotate 90° around Z: (x,y,z) -> (-y,x,z)
        mobile = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        rotation, translation = superpose(fixed, mobile)

        # Apply transformation
        transformed = mobile @ rotation.T + translation
        np.testing.assert_allclose(transformed, fixed, atol=1e-10)

        # Check rotation is proper (det = 1)
        assert np.abs(np.linalg.det(rotation) - 1.0) < 1e-10

    def test_rotation_and_translation(self) -> None:
        """Test combined rotation and translation."""
        fixed = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])

        # Rotate 180° around Z and translate
        mobile = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, 1.0]])
        mobile = mobile + np.array([10.0, 20.0, 30.0])

        rotation, translation = superpose(fixed, mobile)

        transformed = mobile @ rotation.T + translation
        np.testing.assert_allclose(transformed, fixed, atol=1e-10)

    def test_with_weights(self) -> None:
        """Test weighted superposition."""
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, 10.0, 10.0]])
        mobile = fixed.copy()

        # Give last point very high weight, others low weight
        weights = np.array([0.1, 0.1, 0.1, 100.0])

        rotation, translation = superpose(fixed, mobile, weights=weights)

        # Should still get identity since structures are identical
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-10)

    def test_uniform_weights_equals_none(self) -> None:
        """Test that uniform weights give same result as no weights."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

        rot1, trans1 = superpose(fixed, mobile)
        rot2, trans2 = superpose(fixed, mobile, weights=np.ones(3))

        np.testing.assert_allclose(rot1, rot2, atol=1e-10)
        np.testing.assert_allclose(trans1, trans2, atol=1e-10)

    def test_reflection_handling(self) -> None:
        """Test that reflection case is handled (det < 0)."""
        fixed = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Mirror across XY plane: (x,y,z) -> (x,y,-z)
        mobile = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        rotation, _ = superpose(fixed, mobile)

        # Rotation matrix should have det = 1 (no reflection)
        det = np.linalg.det(rotation)
        assert np.abs(det - 1.0) < 1e-10

    def test_shape_mismatch_error(self) -> None:
        """Test error on shape mismatch."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mobile = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            superpose(fixed, mobile)

    def test_wrong_dimensions_error(self) -> None:
        """Test error on wrong number of dimensions."""
        fixed = np.array([[1.0, 2.0], [3.0, 4.0]])
        mobile = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Expected \\(N, 3\\) arrays"):
            superpose(fixed, mobile)

    def test_too_few_points_error(self) -> None:
        """Test error when fewer than 3 points provided."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mobile = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(ValueError, match="at least 3 points"):
            superpose(fixed, mobile)

    def test_weights_shape_mismatch_error(self) -> None:
        """Test error when weights have wrong shape."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = fixed.copy()
        weights = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Weights shape .* incompatible"):
            superpose(fixed, mobile, weights=weights)

    def test_negative_weights_error(self) -> None:
        """Test error on negative weights."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = fixed.copy()
        weights = np.array([1.0, -1.0, 1.0])

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            superpose(fixed, mobile, weights=weights)

    def test_zero_sum_weights_error(self) -> None:
        """Test error when sum of weights is zero."""
        fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        mobile = fixed.copy()
        weights = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Sum of weights must be positive"):
            superpose(fixed, mobile, weights=weights)

    def test_large_structure(self) -> None:
        """Test with larger number of points."""
        np.random.seed(42)
        n_points = 1000

        # Generate random structure
        fixed = np.random.randn(n_points, 3) * 10.0

        # Create mobile by rotating and translating
        angle = np.pi / 6  # 30 degrees
        rot_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        mobile = (fixed @ rot_z.T) + np.array([5.0, -3.0, 2.0])

        rotation, translation = superpose(fixed, mobile)

        # Apply transformation
        transformed = mobile @ rotation.T + translation

        # Should recover original structure
        np.testing.assert_allclose(transformed, fixed, atol=1e-10)
