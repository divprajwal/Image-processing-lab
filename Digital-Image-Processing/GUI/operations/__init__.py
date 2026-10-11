# Auto-discovers operation modules in this package

import importlib
import pkgutil
from pathlib import Path

_registry: dict = {}


def _discover():
    """Scan for modules exposing NAME + process()."""
    package_dir = Path(__file__).resolve().parent
    for finder, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if module_name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__package__}.{module_name}")
        if hasattr(mod, "NAME") and hasattr(mod, "process"):
            _registry[mod.NAME] = mod.process


def get_operations() -> dict:
    """Return the registry dict. Discovers on first call."""
    if not _registry:
        _discover()
    return _registry
