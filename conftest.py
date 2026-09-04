"""Let ``pytest`` find the package when it runs from a clone.

With the ``src/`` layout the package is only importable after
``pip install -e .``.  This adds ``src`` to the path when ``rotorwave`` has not
been installed, so ``pytest`` works straight after unzipping the project.  Once
the package is installed this is a no-op and the installed copy is used.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("rotorwave") is None:  # pragma: no cover - test helper
    source = Path(__file__).resolve().parent / "src"
    if source.is_dir():
        sys.path.insert(0, str(source))
