"""Tests for additional structural similarity metrics."""

import numpy as np
import pytest

from pyprotalign.metrics import gdt, gdt_ha, gdt_ts, tm_score


class TestTMScore:
    """Tests for TM-score."""

    def test_identical_coords(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mobile = fixed.copy()
        assert tm_score(fixed, mobile) == pytest.approx(1.0)

    def test_score_decreases_with_perturbation(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        near = np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [2.0, 0.1, 0.0], [3.0, 0.1, 0.0]])
        far = np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]])

        assert tm_score(fixed, near) > tm_score(fixed, far)

    def test_l_target_changes_normalization(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        mobile = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.0]])

        default_norm = tm_score(fixed, mobile)
        longer_target = tm_score(fixed, mobile, l_target=100)
        assert longer_target < default_norm


class TestGDT:
    """Tests for GDT scores."""

    def test_identical_coords(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mobile = fixed.copy()
        assert gdt_ts(fixed, mobile) == pytest.approx(1.0)
        assert gdt_ha(fixed, mobile) == pytest.approx(1.0)

    def test_gdt_ts_greater_than_or_equal_gdt_ha(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        mobile = np.array([[0.0, 1.5, 0.0], [1.0, 1.5, 0.0], [2.0, 1.5, 0.0], [3.0, 1.5, 0.0]])

        assert gdt_ts(fixed, mobile) >= gdt_ha(fixed, mobile)

    def test_l_target_changes_normalization(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        mobile = np.array([[0.0, 0.2, 0.0], [1.0, 0.2, 0.0], [2.0, 0.2, 0.0], [3.0, 0.2, 0.0]])

        default_norm = gdt_ts(fixed, mobile)
        longer_target = gdt_ts(fixed, mobile, l_target=100)
        assert longer_target < default_norm

    def test_generic_gdt_empty_cutoffs_error(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0]])
        mobile = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="cutoffs cannot be empty"):
            gdt(fixed, mobile, cutoffs=())

    def test_generic_gdt_invalid_l_target(self) -> None:
        fixed = np.array([[0.0, 0.0, 0.0]])
        mobile = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="l_target must be > 0"):
            gdt(fixed, mobile, cutoffs=(1.0,), l_target=0)
