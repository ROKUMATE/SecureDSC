"""SecureDSC: secure wireless semantic communication with adversarial training.

Reproduction of Shi et al. (IEEE Comm. Letters 2025) plus two improvements:
CSI-based neural key generation and an adaptive lambda scheduler.
"""

from .config import Config, load_config, config_from_dict  # noqa: F401

__version__ = "0.1.0"
