from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    raw = Path(config_path).read_bytes()
    # Strip UTF-8 BOM if present
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    # Replace invalid bytes with U+FFFD rather than falling back to latin-1.
    # latin-1 maps 0x80-0x9F to C1 control chars that YAML's SafeLoader rejects.
    # U+FFFD only appears in comments (the ── dividers), so config values are intact.
    text = raw.decode('utf-8', errors='replace')
    return yaml.safe_load(text)
