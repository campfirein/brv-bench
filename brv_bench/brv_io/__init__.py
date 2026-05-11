"""I/O helpers that mirror byterover-cli conventions.

Currently:
- `project_data_dir.resolve_project_data_dir(cwd)`: Python mirror of the CLI's
  `getProjectDataDir(cwd)` path-encoding scheme so the bench can locate a
  project's `query-log/` and `curate-log/` directories without shelling out.
"""

from brv_bench.brv_io.project_data_dir import (
    resolve_project_data_dir,
    sanitize_project_path,
)

__all__ = ("resolve_project_data_dir", "sanitize_project_path")
