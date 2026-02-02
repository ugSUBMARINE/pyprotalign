"""Tests for package initialization."""

import importlib.metadata
from unittest.mock import patch

import pyprotalign


class TestVersionImport:
    """Tests for version import."""

    def test_version_available(self) -> None:
        """Test version is available when package is installed."""
        assert hasattr(pyprotalign, "__version__")
        assert isinstance(pyprotalign.__version__, str)

    def test_version_fallback_on_package_not_found(self) -> None:
        """Test version falls back to 0.0.0 when package not found."""
        import importlib as imp

        with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            # Need to reload module to trigger the except block
            import pyprotalign as pkg

            imp.reload(pkg)
            assert pkg.__version__ == "0.0.0"
