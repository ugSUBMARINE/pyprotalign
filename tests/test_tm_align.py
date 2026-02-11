"""Tests for sequence-independent TM-align implementation."""

import numpy as np

from pyprotalign import tm_align as tm_align_module
from pyprotalign.chain import ProteinChain
from pyprotalign.tm_align import tm_align


def _make_chain(chain_id: str, coords: np.ndarray, sequence_len: int | None = None) -> ProteinChain:
    n = coords.shape[0] if sequence_len is None else sequence_len
    sequence = "A" * n
    if sequence_len is None:
        chain_coords = coords.astype(float)
    else:
        chain_coords = np.full((n, 3), np.nan, dtype=float)
        chain_coords[: coords.shape[0]] = coords
    b = np.ones(n, dtype=float) * 20.0
    occ = np.ones(n, dtype=float)
    return ProteinChain(chain_id=chain_id, sequence=sequence, coords=chain_coords, b_factors=b, occupancies=occ)


class TestTMAlign:
    """Tests for tm_align."""

    def test_identical_backbone(self) -> None:
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.2, 0.0],
                [3.0, 0.0, 0.1],
                [4.5, -0.2, 0.0],
                [6.0, 0.1, -0.1],
            ],
            dtype=float,
        )
        fixed = _make_chain("A", coords)
        mobile = _make_chain("B", coords)

        rotation, translation, tm, num_pairs, mapping = tm_align(fixed, mobile, seeds="single")
        assert np.allclose(rotation, np.eye(3), atol=1e-6)
        assert np.allclose(translation, np.zeros(3), atol=1e-6)
        assert tm > 0.99
        assert num_pairs == len(coords)
        assert mapping == tuple((i, i) for i in range(len(coords)))

    def test_rotation_translation_recovery(self) -> None:
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, -0.5, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.5, 0.0],
            ],
            dtype=float,
        )
        theta = np.deg2rad(35.0)
        r_true = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        t_true = np.array([5.0, -3.0, 2.0], dtype=float)
        mobile_coords = coords @ r_true.T + t_true

        fixed = _make_chain("A", coords)
        mobile = _make_chain("B", mobile_coords)

        rotation, translation, tm, num_pairs, _mapping = tm_align(fixed, mobile, seeds="multi")
        transformed = mobile_coords @ rotation.T + translation
        assert np.allclose(transformed, coords, atol=1e-3)
        assert tm > 0.99
        assert num_pairs >= 5

    def test_partial_overlap_with_extra_mobile_residues(self) -> None:
        core = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        extra = np.array([[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]], dtype=float)
        mobile_coords = np.vstack([extra, core])

        fixed = _make_chain("A", core)
        mobile = _make_chain("B", mobile_coords)

        rotation, translation, tm, num_pairs, mapping = tm_align(fixed, mobile, seeds="multi")
        assert tm > 0.95
        assert num_pairs >= 5
        mapped_mobile_indices = {j for _, j in mapping}
        assert mapped_mobile_indices.issuperset({2, 3, 4, 5, 6})

    def test_sequence_seed_handles_shifted_sequences(self) -> None:
        core = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.2, 0.0],
                [2.0, -0.1, 0.1],
                [3.0, 0.0, -0.1],
                [4.0, 0.1, 0.0],
            ],
            dtype=float,
        )
        extra = np.array([[-10.0, 8.0, 0.0], [-9.0, 8.2, 0.1]], dtype=float)
        mobile_coords = np.vstack([extra, core])

        fixed = _make_chain("A", core)
        mobile = _make_chain("B", mobile_coords)

        _rotation, _translation, tm, num_pairs, mapping = tm_align(fixed, mobile, seeds="sequence")
        assert tm > 0.95
        assert num_pairs >= 5
        mapped_mobile_indices = {j for _, j in mapping}
        assert mapped_mobile_indices.issuperset({2, 3, 4, 5, 6})

    def test_sequence_seed_skips_missing_ca_positions(self) -> None:
        fixed_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        mobile_coords = fixed_coords.copy()

        fixed = _make_chain("A", fixed_coords, sequence_len=7)
        mobile = _make_chain("B", mobile_coords, sequence_len=7)

        _rotation, _translation, tm, num_pairs, mapping = tm_align(fixed, mobile, seeds="sequence")
        assert np.isclose(tm, 5.0 / 7.0, atol=1e-6)
        assert num_pairs == 5
        assert mapping == tuple((i, i) for i in range(5))

    def test_secondary_seed_handles_shifted_sequences(self) -> None:
        core = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.2, 0.0],
                [2.0, -0.1, 0.1],
                [3.0, 0.0, -0.1],
                [4.0, 0.1, 0.0],
            ],
            dtype=float,
        )
        extra = np.array([[-10.0, 8.0, 0.0], [-9.0, 8.2, 0.1]], dtype=float)
        mobile_coords = np.vstack([extra, core])

        fixed = _make_chain("A", core)
        mobile = _make_chain("B", mobile_coords)

        _rotation, _translation, tm, num_pairs, mapping = tm_align(fixed, mobile, seeds="secondary")
        assert tm > 0.95
        assert num_pairs >= 5
        mapped_mobile_indices = {j for _, j in mapping}
        assert mapped_mobile_indices.issuperset({2, 3, 4, 5, 6})

    def test_secondary_seed_fallback_to_multi_when_sparse(self, monkeypatch: object) -> None:
        def _fake_secondary_seed(_fixed: np.ndarray, _mobile: np.ndarray) -> list[tuple[int, int]]:
            return []

        monkeypatch.setattr(tm_align_module, "_build_secondary_seed_mapping", _fake_secondary_seed)

        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.2, 0.0],
                [3.0, 0.0, 0.1],
                [4.5, -0.2, 0.0],
                [6.0, 0.1, -0.1],
            ],
            dtype=float,
        )
        fixed = _make_chain("A", coords)
        mobile = _make_chain("B", coords)

        _rotation, _translation, tm, num_pairs, mapping = tm_align(fixed, mobile, seeds="secondary")
        assert tm > 0.99
        assert num_pairs == len(coords)
        assert mapping == tuple((i, i) for i in range(len(coords)))
