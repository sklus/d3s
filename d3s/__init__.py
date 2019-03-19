# -*- coding: utf-8 -*-
"""
Data-driven analysis of dynamical systems and associated transfer operators.

This package contains methods such as:
  - DMD/TICA
  - EDMD
  - Ulam's method
  - SINDy
"""

# TODO: import api functions here
from .algorithms import (
    amuse,
    sindy,
    dmd,
    edmd,
    kedmd,
    ulam,
    tica,
)
