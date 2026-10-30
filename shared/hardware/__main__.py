"""Write the environment manifests: ``python -m shared.hardware``."""

from shared.hardware.probe import write_environment_manifests

if __name__ == "__main__":
    write_environment_manifests()
