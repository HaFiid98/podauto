"""Minimal .env loader.

providers.json records only the NAME of the environment variable holding each
key, which is what keeps that file safe to commit. That design needs the values
to actually be in the environment, so something has to put them there -- and
"remember to export three variables before every run" is a step that gets
skipped.

No python-dotenv dependency: the format we need is `KEY=value` with `#`
comments, and a real parser would be more surface than the job deserves.

Precedence: an already-exported variable always wins. CI and shell exports are
more deliberate than a file, and silently overriding them would make the
override invisible.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILE = ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> list[str]:
    """Set any variable named in `path` that is not already in the environment.

    Returns the names that were loaded -- names only, never values, because the
    caller's natural next move is to log the result.
    Missing file is not an error: the environment may already be populated.
    """
    p = Path(path)
    if not p.is_file():
        return []

    loaded: list[str] = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded
