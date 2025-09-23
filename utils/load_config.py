import argparse, importlib.util, inspect, os, sys
from types import ModuleType

# Import your base class so we can detect subclasses
from .config import *
from .model import *

def import_module_from_path(path: str) -> ModuleType:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    name = os.path.splitext(os.path.basename(path))[0]  # e.g. "two_tets"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def pick_config_class(mod: ModuleType, explicit: str | None) -> type[SimulationConfig]:
    # If user names the class, use it.
    if explicit:
        obj = getattr(mod, explicit, None)
        if obj is None or not inspect.isclass(obj) or not issubclass(obj, SimulationConfig):
            raise ValueError(f"'{explicit}' is not a SimulationConfig in {mod.__name__}")
        return obj

    # Otherwise, auto-pick:
    # 1) CONFIG if provided (instance or class)
    if hasattr(mod, "CONFIG"):
        cfg = getattr(mod, "CONFIG")
        if inspect.isclass(cfg) and issubclass(cfg, SimulationConfig):
            return cfg
        if isinstance(cfg, SimulationConfig):
            # Wrap instance in a trivial class so caller constructs consistently
            class _ProvidedConfig(type(cfg)):  # type: ignore[misc, valid-type]
                pass
            return _ProvidedConfig  # will call __init__ with no args; if that fails, user should pass --class
        # else ignore and fall through

    # 2) If there is exactly one subclass of SimulationConfig defined in this module, use it.
    candidates = []
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and issubclass(obj, SimulationConfig):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]

    names = ", ".join(c.__name__ for c in candidates) or "(none found)"
    raise ValueError(
        f"Ambiguous or missing config class in {mod.__name__}. "
        f"Found: {names}. Specify one with --class CLASSNAME or export CONFIG."
    )

def load_config(path: str)->SimulationConfig:
    mod = import_module_from_path(path)
    return pick_config_class(mod, None)() #construct an instance of the config and return it