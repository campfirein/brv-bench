"""Resolve the per-project data directory for a byterover-cli working dir.

Python mirror of byterover-cli's `getProjectDataDir(cwd)` /
`sanitizeProjectPath(resolved)` / `getGlobalDataDir()` chain so the bench
can locate a project's `query-log/` and `curate-log/` JSON files for
telemetry consumption without shelling out to the CLI.

The encoding scheme is documented in
`byterover-cli/src/server/utils/path-utils.ts`. Any drift between the
CLI's implementation and this mirror means the bench reads from the
wrong directory and reports `"unknown"` for every telemetry field — keep
the two in lockstep.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
from pathlib import Path

#: Characters illegal in Windows directory names, mapped to their
#: percent-encoded forms. Must match WINDOWS_ILLEGAL_CHARS in
#: byterover-cli/src/server/utils/path-utils.ts exactly.
_WINDOWS_ILLEGAL_CHARS: dict[str, str] = {
    '"': "%22",
    "*": "%2A",
    ":": "%3A",
    "<": "%3C",
    ">": "%3E",
    "?": "%3F",
    "|": "%7C",
}

#: Cap on the sanitized directory name length. Beyond this we truncate
#: and append a hash suffix to preserve uniqueness.
_MAX_SANITIZED_LENGTH = 200

#: Length of the hex hash suffix appended to truncated names.
_HASH_SUFFIX_LENGTH = 12

#: brv's data-dir name under the platform's user-data root.
_GLOBAL_DATA_DIR = "brv"

#: Subdirectory holding per-project state under the global data dir.
_GLOBAL_PROJECTS_DIR = "projects"

#: Windows drive-letter prefix (e.g. `C:\foo` → strip the colon → `C\foo`).
_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):")


def sanitize_project_path(resolved_path: str) -> str:
    """Encode a resolved absolute path into a safe, collision-free name.

    Mirrors `sanitizeProjectPath` in path-utils.ts. Output must be
    byte-identical to the CLI for telemetry to land in the same dir.

    Args:
        resolved_path: An absolute path (output of `Path.resolve()`).
            Must have at least one non-separator component for a usable
            result; root paths (`/`) produce `""`.

    Returns:
        Safe directory name for `<projects-dir>/<name>/`.
    """
    # Strip Windows drive colon (C:\foo → C\foo).
    normalized = _WIN_DRIVE_RE.sub(r"\1", resolved_path)

    # Split on / and \, drop empty components.
    components = [c for c in re.split(r"[/\\]+", normalized) if c]

    encoded: list[str] = []
    for component in components:
        # Order matters: % must encode FIRST so we never double-encode the
        # percent sign we're about to introduce as the encoding marker.
        result = component.replace("%", "%25").replace("--", "%2D%2D")
        for char, replacement in _WINDOWS_ILLEGAL_CHARS.items():
            result = result.replace(char, replacement)
        encoded.append(result)

    joined = "--".join(encoded)

    if len(joined) <= _MAX_SANITIZED_LENGTH:
        return joined

    # Truncate + hash. Use the FULL joined string as hash input so
    # different long paths produce different suffixes. `---` triple
    # dash is unambiguous because `--` inside components has been
    # encoded to `%2D%2D`.
    digest = hashlib.sha256(joined.encode()).hexdigest()[:_HASH_SUFFIX_LENGTH]
    prefix_length = _MAX_SANITIZED_LENGTH - _HASH_SUFFIX_LENGTH - 3
    return joined[:prefix_length] + "---" + digest


def _global_data_dir() -> Path:
    """Resolve brv's per-user data directory by platform.

    Mirrors `getGlobalDataDir()` in global-data-path.ts:
      - `BRV_DATA_DIR` env var trumps everything
      - macOS: `~/Library/Application Support/brv`
      - Windows: `%LOCALAPPDATA%/brv` (falls back to `~/AppData/Local/brv`)
      - Linux: `$XDG_DATA_HOME/brv` if set, else `~/.local/share/brv`
    """
    override = os.environ.get("BRV_DATA_DIR")
    if override:
        return Path(override)

    system = platform.system()

    if system == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / _GLOBAL_DATA_DIR
        return Path.home() / "AppData" / "Local" / _GLOBAL_DATA_DIR

    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / _GLOBAL_DATA_DIR

    # Linux (and any other Unix-like): respect XDG_DATA_HOME if set.
    if system == "Linux":
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            return Path(xdg) / _GLOBAL_DATA_DIR

    return Path.home() / ".local" / "share" / _GLOBAL_DATA_DIR


def resolve_project_data_dir(cwd: str | Path) -> str:
    """Map a working directory to brv's per-project data directory.

    Returns the absolute path `<global-data-dir>/projects/<sanitized>/`
    where `<sanitized>` is produced by `sanitize_project_path()` on the
    realpath of `cwd`. This is where brv writes `query-log/*.json`,
    `curate-log/*.json`, `keystore/`, `blobs/`, and `task-history/` for
    the project rooted at `cwd`.

    Args:
        cwd: The working directory of the brv project. Symlinks are
            resolved before sanitization, so two distinct paths that
            point at the same project produce the same data-dir.

    Returns:
        Absolute path string (without trailing separator).
    """
    resolved = str(Path(cwd).resolve())
    sanitized = sanitize_project_path(resolved)
    return str(_global_data_dir() / _GLOBAL_PROJECTS_DIR / sanitized)
