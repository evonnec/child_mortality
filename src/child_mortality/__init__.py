"""Ratio metrics for every dimensional cut of a table."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("child-mortality")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

from child_mortality.cube import ratio_cube
from child_mortality.survey import cd_ceb_cube

__all__ = ["__version__", "cd_ceb_cube", "ratio_cube"]
