"""Human-readable byte counts.

A report can hold a 4-byte empty delta frame and a 2 GB raw baseline on adjacent lines. Ten
orders of magnitude in one table needs consistent rendering.
"""

from __future__ import annotations

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def human_bytes(count: float) -> str:
    """``human_bytes(1536)`` -> ``'1.50 KB'``. Whole bytes print without decimals."""
    value = float(count)
    for unit in _UNITS:
        if abs(value) < 1024 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")
