"""Repository-local import shim for the src-layout package.

This lets `python -m election_sim.cli` work from a fresh checkout without
requiring an editable install first.
"""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "election_sim"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]
