"""Curate command — populate context tree from a benchmark dataset."""

import asyncio
import logging
from dataclasses import dataclass

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
) -> CurateResult:
    """Curate a single corpus document via brv CLI.

    Formats the document using the prompt template, then runs:
        brv curate <formatted_content> --headless --format json
    """
    formatted = prompt_config.curate_template.format(
        doc_id=doc.doc_id,
        source=doc.source,
        content=doc.content,
    )

    proc = await asyncio.create_subprocess_exec(
        "brv",
        "curate",
        formatted,
        "--detach",
        "--format",
        "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
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
) -> CurateSummary:
    """Run the full curation pipeline.

    Iterates over corpus documents sequentially, formatting each
    with the prompt template and passing to `brv curate`.

    Args:
        corpus: Corpus documents from the benchmark dataset.
        prompt_config: Dataset-specific prompt templates.

    Returns:
        CurateSummary with per-document results.
    """
    if not corpus:
        return CurateSummary(total=0, succeeded=0, failed=0, results=())

    results: list[CurateResult] = []
    for doc in tqdm(corpus, desc="Curating", unit="doc"):
        result = await curate_doc(doc, prompt_config)
        results.append(result)

    succeeded = sum(1 for r in results if r.success)
    return CurateSummary(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=tuple(results),
    )
