"""Curate command — populate context tree from a benchmark dataset."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from brv_bench.types import CorpusDocument, PromptConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurateResult:
    """Result of a single curate operation."""

    doc_id: str
    success: bool
    message: str


@dataclass(frozen=True)
class CurateSummary:
    """Summary of the full curate run."""

    total: int
    succeeded: int
    failed: int
    results: tuple[CurateResult, ...]


async def curate_doc(
    doc: CorpusDocument,
    prompt_config: PromptConfig,
    brv_bin: str = "brv",
    cwd: Path | None = None,
) -> CurateResult:
    """Curate a single corpus document via brv CLI.

    Formats the document using the prompt template, then runs:
        <brv_bin> curate <formatted_content> --detach --format json

    For two-config A/B orchestration each config's checkout provides its
    own `bin/run.js`; `brv_bin` and `cwd` let the caller pin this curate
    invocation to one specific checkout. Defaults preserve the
    single-config flow (`brv` on PATH, bench's cwd).
    """
    formatted = prompt_config.curate_template.format(
        doc_id=doc.doc_id,
        source=doc.source,
        content=doc.content,
    )

    proc = await asyncio.create_subprocess_exec(
        brv_bin,
        "curate",
        formatted,
        "--detach",
        "--format",
        "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode == 0:
        return CurateResult(
            doc_id=doc.doc_id,
            success=True,
            message=stdout.decode().strip(),
        )
    else:
        msg = stderr.decode().strip() or stdout.decode().strip()
        return CurateResult(
            doc_id=doc.doc_id,
            success=False,
            message=msg,
        )


async def curate(
    corpus: tuple[CorpusDocument, ...],
    prompt_config: PromptConfig,
    brv_bin: str = "brv",
    cwd: Path | None = None,
) -> CurateSummary:
    """Run the full curation pipeline.

    Iterates over corpus documents sequentially, formatting each
    with the prompt template and passing to `brv curate`.

    Args:
        corpus: Corpus documents from the benchmark dataset.
        prompt_config: Dataset-specific prompt templates.
        brv_bin: Path to the brv binary (default `"brv"` from PATH). For
            two-config orchestration, set to `<checkout>/bin/run.js` so
            each config's curates fork their own daemon from their own
            checkout's `dist/`.
        cwd: Working directory for the brv subprocess (default: inherit
            from bench). The brv CLI's `.brv/` project resolution keys
            off cwd, so this is what isolates each config's context tree.

    Returns:
        CurateSummary with per-document results.
    """
    if not corpus:
        return CurateSummary(total=0, succeeded=0, failed=0, results=())

    results: list[CurateResult] = []
    for doc in tqdm(corpus, desc="Curating", unit="doc"):
        result = await curate_doc(doc, prompt_config, brv_bin=brv_bin, cwd=cwd)
        results.append(result)

    succeeded = sum(1 for r in results if r.success)
    return CurateSummary(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=tuple(results),
    )
