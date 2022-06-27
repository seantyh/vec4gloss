from hashlib import sha1
from pathlib import Path
from typing import List

def check_hashes(paths: List[Path]):
    hash_res = {}
    if not isinstance(paths, list):
        paths = [paths]

    for path_x in paths:
        path_x = Path(path_x)
        h = sha1()
        h.update(path_x.read_bytes())
        m = h.hexdigest()[:6]
        hash_res[str(path_x)] = m
        print(path_x, m)
    return hash_res

