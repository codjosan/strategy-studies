from pathlib import Path

import pytest

from hlmicro.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session")
def capture_dir(tmp_path_factory) -> Path:
    """Two days of synthetic capture data shared by the test session."""
    return generate(tmp_path_factory.mktemp("capture"), SyntheticConfig(days=2, seed=11))
