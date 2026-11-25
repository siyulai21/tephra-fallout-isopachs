"""Joint directional thinning with radially evolving fan orientation and correlated local departures."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@lru_cache(maxsize=2)
def engine_sha256(*, observation_source=None):
    """Identify the numerical field, likelihood, projection and common partition rules."""
    here = Path(__file__).resolve().parent
    framework = here.parents[2]
    files = [
        here / (name + ".py")
        for name in ("geometry", "spread", "conditioning", "fit", "posterior", "integrals", "validation")
    ]
    files += [
        framework / name
        for name in (
            "core/observations.py",
            "core/geometry.py",
            "core/validation.py",
            "core/likelihood.py",
            "core/distributions.py",
            "models/directional/geometry.py",
            "models/directional/ensemble/members.py",
            "models/directional/ensemble/integrals.py",
            "models/mass_mixture/geometry.py",
        )
    ]
    if observation_source is not None:
        files[files.index(framework / "core/observations.py")] = Path(observation_source)
    files += [framework.parent / "probabilistic_isopach/global_volume.py"]
    return sha256(b"".join(sha256(file.read_bytes()).digest() for file in files)).hexdigest()
