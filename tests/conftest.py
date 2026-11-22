from pathlib import Path

import pytest

from hlmicro.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session", autouse=True)
def isolated_cache(tmp_path_factory):
    """Loader cache in a temporary directory, never the user's ~/.cache/hlmicro."""
    mp = pytest.MonkeyPatch()
    mp.setenv("HLMICRO_CACHE_DIR", str(tmp_path_factory.mktemp("hlmicro_cache")))
    mp.delenv("HLMICRO_NO_CACHE", raising=False)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def capture_dir(tmp_path_factory) -> Path:
    """Two days of synthetic capture data shared by the test session."""
    return generate(tmp_path_factory.mktemp("capture"), SyntheticConfig(days=2, seed=11))
