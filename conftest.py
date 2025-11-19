# conftest.py
# Pytest fixtures / env helpers for project (dataset path, quick sample loader)
import os
import pytest
from pathlib import Path

DEFAULT_DATA_ROOT = os.environ.get("MALARIA_DATA_ROOT", "data/cell_images")

@pytest.fixture(scope="session")
def data_root():
    """Return the dataset root path used by tests and quick scripts."""
    root = Path(DEFAULT_DATA_ROOT)
    if not root.exists():
        pytest.skip(f"Dataset root '{root}' does not exist. Set MALARIA_DATA_ROOT or create the folder.")
    return root

@pytest.fixture(scope="session")
def sample_image_path(data_root):
    """Return a single sample image path for smoke tests."""
    parasitized = list((data_root / "Parasitized").glob("*"))
    uninfected = list((data_root / "Uninfected").glob("*"))
    sample = (parasitized + uninfected)
    if not sample:
        pytest.skip("No images found in dataset folders.")
    return sample[0]

def pytest_addoption(parser):
    parser.addoption("--data-root", action="store", default=DEFAULT_DATA_ROOT,
                     help="Path to malaria dataset root containing Parasitized/ and Uninfected/ subfolders")
