"""
marianas_tools
==============

Utilities for handling Marianas microscope data, including
automatic assembly of Z/T/C image stacks into OME-TIFF.

Author: Timothy J. Stasevich
"""

from importlib.metadata import version, PackageNotFoundError

# Expose high-level API if desired
# (optional — keeps import surface clean)
from .assemble_stacks_auto import main as assemble_stacks

try:
    __version__ = version("marianas-tools")
except PackageNotFoundError:
    # Package not installed (e.g., running from source)
    __version__ = "0.0.0-dev"

__all__ = ["assemble_stacks", "__version__"]