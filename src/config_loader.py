import yaml
import os

def load_config(config_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_path(config, key_path, default=None):
    """Helper to get nested keys from config, returning default if not found."""
    keys = key_path.split('.')
    val = config
    try:
        for key in keys:
            val = val[key]
        return val
    except KeyError:
        return default
    except TypeError:
        return default