"""star_ea – Event activity analysis for pp collisions at STAR."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("star-ea")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "simulation",
    "observables",
    "multifold",
    "plotting",
    "io",
]
