"""Unit conversion helpers.

The public API of :mod:`rotorwave` takes **frequencies in Hz** and **spin
speeds in rpm**, because that is how rotordynamic results are reported, and
converts to SI internally.  Every argument name carries its unit as a suffix
(``frequency_hz``, ``spin_rpm``) so that no conversion is ever implicit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rpm_to_rad", "rad_to_rpm", "hz_to_rad", "rad_to_hz"]

_TWO_PI = 2.0 * np.pi

Number = float | np.ndarray


def rpm_to_rad(speed_rpm: Number) -> Number:
    """Convert a rotating speed from rpm to rad/s."""
    return speed_rpm * _TWO_PI / 60.0


def rad_to_rpm(speed_rad: Number) -> Number:
    """Convert a rotating speed from rad/s to rpm."""
    return speed_rad * 60.0 / _TWO_PI


def hz_to_rad(frequency_hz: Number) -> Number:
    """Convert a frequency from Hz to rad/s."""
    return frequency_hz * _TWO_PI


def rad_to_hz(omega_rad: Number) -> Number:
    """Convert an angular frequency from rad/s to Hz."""
    return omega_rad / _TWO_PI
