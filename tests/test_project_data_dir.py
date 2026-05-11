"""Tests for `resolve_project_data_dir` — the Python mirror of byterover-cli's
`getProjectDataDir(cwd)` path-encoding scheme.

The function must produce paths byte-identical to what the CLI writes
telemetry to, otherwise the bench's telemetry consumer reads from the
wrong directory and reports `"unknown"` for every field.

Reference implementation lives at
`byterover-cli/src/server/utils/{path-utils,global-data-path}.ts`.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from brv_bench.brv_io import resolve_project_data_dir, sanitize_project_path


class TestSanitizeProjectPath:
    """`sanitize_project_path` splits on path separators, percent-encodes
    `%` and `--` and Windows-illegal chars within each component, joins
    components with `--`. Output > 200 chars is truncated and hash-suffixed
    so collision-freeness is preserved."""

    def test_simple_unix_path(self):
        assert sanitize_project_path("/home/user/project") == "home--user--project"

    def test_strips_windows_drive_colon(self):
        assert sanitize_project_path("C:/Users/Phat/project") == "C--Users--Phat--project"

    def test_strips_backslash_drive(self):
        assert sanitize_project_path("C:\\Users\\Phat\\project") == "C--Users--Phat--project"

    def test_percent_encodes_percent_sign(self):
        # `%` must encode first so we don't double-encode the encoding marker
        assert sanitize_project_path("/foo/100%bar") == "foo--100%25bar"

    def test_percent_encodes_double_dash_in_component(self):
        # A component containing `--` would otherwise collide with the join
        # separator. Encode it so the resulting string is unambiguously
        # parseable back to components.
        assert sanitize_project_path("/foo/double--dash/baz") == "foo--double%2D%2Ddash--baz"

    def test_percent_encodes_windows_illegal_chars(self):
        # The CLI encodes " * : < > ? | as their percent-encoded forms so
        # the directory name is valid on Windows. Verify on a Unix path
        # that mentions them in a component.
        assert sanitize_project_path("/foo/has:colon") == "foo--has%3Acolon"
        assert sanitize_project_path("/foo/star*") == "foo--star%2A"
        assert sanitize_project_path("/foo/quote\"x") == "foo--quote%22x"

    def test_root_path_returns_empty_string(self):
        # Single slash → no non-separator components → empty string.
        # The CLI accepts this output but flags it as invalid for project
        # registration. We mirror the output exactly.
        assert sanitize_project_path("/") == ""

    def test_long_path_is_truncated_with_hash_suffix(self):
        # Path with sanitized form > 200 chars must be truncated to
        # (200 - 12 - 3) = 185 chars + `---` + first 12 hex chars of
        # sha256(joined). The hash uses the FULL joined string as input
        # so different long paths produce different hashes.
        long_component = "a" * 100
        path = f"/{long_component}/{long_component}/{long_component}"
        result = sanitize_project_path(path)
        # Total joined length: 100 + 2 + 100 + 2 + 100 = 304 chars → must truncate.
        assert len(result) == 200
        assert "---" in result
        # Suffix is 12 hex chars of sha256(joined)
        joined_full = "--".join([long_component, long_component, long_component])
        expected_hash = hashlib.sha256(joined_full.encode()).hexdigest()[:12]
        assert result.endswith("---" + expected_hash)
        # Prefix preserved
        assert result.startswith(joined_full[:185])

    def test_path_at_boundary_is_not_truncated(self):
        # Exactly 200 chars after join → no truncation.
        comp = "a" * 99
        joined = sanitize_project_path(f"/{comp}/{comp}")  # 99 + 2 + 99 = 200
        assert len(joined) == 200
        assert "---" not in joined


class TestResolveProjectDataDir:
    """`resolve_project_data_dir(cwd)` → `<global-data-dir>/projects/<sanitized>`
    where `<global-data-dir>` follows the same per-OS conventions as
    byterover-cli's `getGlobalDataDir()`."""

    def test_macos_path(self, tmp_path: Path):
        with patch("platform.system", return_value="Darwin"), \
             patch.dict(os.environ, {}, clear=True), \
             patch("pathlib.Path.home", return_value=Path("/Users/test")):
            result = resolve_project_data_dir(str(tmp_path))
            expected_prefix = "/Users/test/Library/Application Support/brv/projects/"
            assert result.startswith(expected_prefix), \
                f"expected prefix {expected_prefix!r}, got {result!r}"

    def test_linux_default_path(self, tmp_path: Path):
        with patch("platform.system", return_value="Linux"), \
             patch.dict(os.environ, {}, clear=True), \
             patch("pathlib.Path.home", return_value=Path("/home/test")):
            result = resolve_project_data_dir(str(tmp_path))
            expected_prefix = "/home/test/.local/share/brv/projects/"
            assert result.startswith(expected_prefix)

    def test_linux_xdg_data_home_override(self, tmp_path: Path):
        with patch("platform.system", return_value="Linux"), \
             patch.dict(os.environ, {"XDG_DATA_HOME": "/custom/xdg"}, clear=True):
            result = resolve_project_data_dir(str(tmp_path))
            assert result.startswith("/custom/xdg/brv/projects/")

    def test_brv_data_dir_env_override_trumps_platform_default(self, tmp_path: Path):
        # BRV_DATA_DIR replaces the entire global data dir.
        with patch.dict(os.environ, {"BRV_DATA_DIR": "/override/brv"}, clear=True):
            result = resolve_project_data_dir(str(tmp_path))
            assert result.startswith("/override/brv/projects/")

    def test_sanitized_component_matches_cli_format(self, tmp_path: Path):
        # Use BRV_DATA_DIR override to make the test platform-independent.
        with patch.dict(os.environ, {"BRV_DATA_DIR": "/override/brv"}, clear=True):
            real = str(tmp_path.resolve())
            result = resolve_project_data_dir(real)
            sanitized = sanitize_project_path(real)
            assert result == f"/override/brv/projects/{sanitized}"

    def test_realpath_resolves_symlinks(self, tmp_path: Path):
        # The CLI calls realpath(cwd) before sanitizing. Two paths that
        # resolve to the same target produce identical data-dir paths
        # (load-bearing for telemetry to land where the daemon writes it).
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)

        with patch.dict(os.environ, {"BRV_DATA_DIR": "/override/brv"}, clear=True):
            via_link = resolve_project_data_dir(str(link))
            via_real = resolve_project_data_dir(str(target))
            assert via_link == via_real
