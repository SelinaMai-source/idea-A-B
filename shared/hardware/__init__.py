"""Local environment probes.

Imports are lazy so ``python -m shared.hardware.probe`` is not preloaded by
this package init.
"""

__all__ = ["collect_environment", "write_environment_manifests"]


def __getattr__(name: str):
    if name in __all__:
        from shared.hardware import probe

        return getattr(probe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
