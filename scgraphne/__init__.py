import importlib

from .data import preprocess
from .run_scgraphne import run_scgraphne
from .utils import setup_seed, sample, read_data

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution

    version = lambda name: get_distribution(name).version

__name__ = "scGraphNE"
try:
    __version__ = version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = '0.1.1'
