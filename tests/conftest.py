import pytest
import os


_root_dir = f'{os.path.dirname(__file__)}/..'
_plot_dir = f'{_root_dir}/tests/plots'
_data_dir = f'{_root_dir}/example/data'


@pytest.fixture
def file_optical():
    spectrum_path = f'{_data_dir}/SN2006mo.dat'
    return spectrum_path


@pytest.fixture
def file_NIR():
    spectrum_path = f'{_data_dir}/SN2011fe.dat'
    return spectrum_path


@pytest.fixture
def plot_dir():
    return _plot_dir


@pytest.fixture
def can_plot():
    return not os.getenv('CI')


def pytest_sessionstart(session):
    os.makedirs(_plot_dir, exist_ok=True)
