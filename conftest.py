import pytest
import os


@pytest.fixture
def sample_file():
    data_dir = f'{os.path.dirname(__file__)}/example/data'
    spectrum_path = f'{data_dir}/SN2006mo.dat'
    return spectrum_path
