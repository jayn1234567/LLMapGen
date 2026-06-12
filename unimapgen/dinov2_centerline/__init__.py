"""Minimal DINOv2 -> Qwen centerline prediction route.

This package is the cleaned, recommended entry for the earlier DINOv2 route.
It keeps only the pieces needed for RC road-centerline JSON SFT and inference:

- trainroot preparation with Douglas simplification and fragment merge
- DINOv2 visual-token bridge model
- one training entry and one prediction entry
"""

from __future__ import annotations

__all__ = []

