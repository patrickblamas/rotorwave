"""Allow the examples to run from a clone without installing the package.

Importing this module puts ``../src`` on ``sys.path`` when ``rotorwave`` has not
been installed, so every example can be opened in Spyder or a notebook and run
straight away.  Once the package is installed (``pip install -e .``) this is a
no-op.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("rotorwave") is None:  # pragma: no cover - dev helper
    source = Path(__file__).resolve().parent.parent / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
