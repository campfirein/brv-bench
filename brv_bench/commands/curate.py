"""Curate command — populate context tree from source files."""

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CurateResult:
    """Result of a single curate operation."""

    file: Path
    success: bool
    message: str


@dataclass(frozen=True)
class CurateSummary:
    """Summary of the full curate run."""

    total: int
    succeeded: int
    failed: int
    results: tuple[CurateResult, ...]


async def curate_file(file: Path) -> CurateResult:
    """Curate a single file via brv CLI.

    Runs: brv curate --headless -f <file>
    """
    proc = await asyncio.create_subprocess_exec(
        "brv",
        "curate",
        "--headless",
        "-f",
        str(file),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode == 0:
        return CurateResult(file=file, success=True, message=stdout.decode().strip())
    else:
        msg = stderr.decode().strip() or stdout.decode().strip()
        return CurateResult(file=file, success=False, message=msg)


def collect_source_files(source_dir: Path) -> list[Path]:
    """Collect all files from the source directory, sorted for determinism."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    files = [f for f in source_dir.rglob("*") if f.is_file()]
    return sorted(files)


async def curate(source_dir: Path) -> CurateSummary:
    """Run the full curation pipeline.

    Walks source_dir, calls `brv curate --headless -f <file>` for each file,
    and returns a summary of results.

    Args:
        source_dir: Directory containing source files to curate.

    Returns:
        CurateSummary with per-file results.
    """
    files = collect_source_files(source_dir)

    if not files:
        return CurateSummary(total=0, succeeded=0, failed=0, results=())

    results: list[CurateResult] = []
    for file in files:
        result = await curate_file(file)
        results.append(result)

    succeeded = sum(1 for r in results if r.success)
    return CurateSummary(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=tuple(results),
    )
