from pathlib import Path
import sys
import yaml

def load_config(filename=None):
    filename = filename or sys.argv[1]
    path = Path(filename).expanduser().resolve()
    with path.open() as f:
        config = yaml.safe_load(f)
    config['_root'] = str(Path(__file__).resolve().parents[3])
    config['_config_path'] = str(path)
    return config

def project_path(config, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config['_root']) / path
