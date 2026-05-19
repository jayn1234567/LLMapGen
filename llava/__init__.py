"""Legacy compatibility package.

The active framework namespace is ``mllm``.  This shim keeps old
``llava.*`` imports and older command lines importable while new code uses
``mllm.*`` directly.
"""

from __future__ import annotations

import importlib
import sys

_mllm = importlib.import_module("mllm")

__path__ = _mllm.__path__
__all__ = getattr(_mllm, "__all__", [])

for _name in dir(_mllm):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_mllm, _name)

sys.modules[__name__] = _mllm
