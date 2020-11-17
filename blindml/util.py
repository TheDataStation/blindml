import hashlib
import json
from typing import Dict, Any


def dict_hash(dictionary: Dict[str, Any], omit_keys=None) -> str:
    if omit_keys is None:
        omit_keys = []

    for key in omit_keys:
        dictionary.pop(key)
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
