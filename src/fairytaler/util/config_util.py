import os

from typing import Any, Dict

__all__ = ["load_config"]

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Loads configuration from a .json or .yaml file.

    :param config_file: Path to the configuration file.
    :return: Dictionary containing the configuration.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if config_file.endswith(".json"):
        import json
        with open(config_file, "r") as f:
            return json.load(f) # type: ignore[no-any-return]
    elif config_file.endswith(".yaml"):
        import yaml
        with open(config_file, "r") as f:
            return yaml.safe_load(f) # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unsupported config file format: {config_file}")
