"""LoCoMo dataset transformer.

Transforms LoCoMo's raw JSON (locomo10.json) into brv-bench's
canonical BenchmarkDataset format.

LoCoMo structure:
- 10 conversations, each with multiple sessions
- Each session has dialog turns with dia_ids (e.g., "D1:3")
- Observations: per-session, per-speaker extracted facts
  linked to dialog turns
- QA: questions with evidence pointing to dia_ids

Corpus strategy:
- Each observation set (session + speaker) becomes one
  CorpusDocument. This gives fine-grained retrieval targets.

Evidence mapping:
- QA evidence dia_ids are mapped to observation doc_ids
  via the dia_id links in observations.
- If a dia_id has no observation, we fall back to the
  session-level: all observation docs from that session
  are considered relevant.
"""

import json
from pathlib import Path

from brv_bench.types import (
    BenchmarkDataset,
    CorpusDocument,
    GroundTruthEntry,
)

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "commonsense",
    5: "adversarial",
}


def transform(raw_path: Path) -> BenchmarkDataset:
    """Transform LoCoMo JSON into BenchmarkDataset.

    Args:
        raw_path: Path to locomo10.json.

    Returns:
        BenchmarkDataset with observation-based corpus
        and QA-based entries.
    """
    with open(raw_path) as f:
        raw_data = json.load(f)

    all_corpus: list[CorpusDocument] = []
    all_entries: list[GroundTruthEntry] = []

    for sample in raw_data:
        sample_id = sample["sample_id"]
        corpus, dia_to_docs, session_docs = (
            _build_corpus(sample_id, sample["observation"])
        )
        all_corpus.extend(corpus)

        entries = _build_entries(
            sample["qa"], dia_to_docs, session_docs
        )
        all_entries.extend(entries)

    return BenchmarkDataset(
        name="locomo",
        corpus=tuple(all_corpus),
        entries=tuple(all_entries),
    )


def _build_corpus(
    sample_id: str,
    observations: dict,
) -> tuple[
    list[CorpusDocument],
    dict[str, set[str]],
    dict[int, set[str]],
]:
    """Build corpus documents from observations.

    Returns:
        - List of CorpusDocuments
        - Mapping: dia_id -> set of doc_ids
        - Mapping: session_number -> set of doc_ids
    """
    corpus: list[CorpusDocument] = []
    dia_to_docs: dict[str, set[str]] = {}
    session_docs: dict[int, set[str]] = {}

    for obs_key, speakers in observations.items():
        session_num = int(
            obs_key.replace("session_", "")
            .replace("_observation", "")
        )

        if session_num not in session_docs:
            session_docs[session_num] = set()

        for speaker, entries in speakers.items():
            doc_id = (
                f"{sample_id}_s{session_num}"
                f"_{speaker.lower()}"
            )
            lines: list[str] = []

            for entry in entries:
                text = entry[0]
                raw_ids = entry[1]
                lines.append(text)

                # dia_id can be str or list[str]
                if isinstance(raw_ids, str):
                    raw_ids = [raw_ids]
                for dia_id in raw_ids:
                    if dia_id not in dia_to_docs:
                        dia_to_docs[dia_id] = set()
                    dia_to_docs[dia_id].add(doc_id)

            content = "\n".join(lines)
            corpus.append(
                CorpusDocument(
                    doc_id=doc_id,
                    content=content,
                    source=f"session_{session_num}",
                )
            )
            session_docs[session_num].add(doc_id)

    return corpus, dia_to_docs, session_docs


def _extract_session_number(dia_id: str) -> int | None:
    """Extract session number from dia_id.

    Format: 'D<session>:<turn>' e.g., 'D1:3' -> 1
    """
    try:
        prefix = dia_id.split(":")[0]
        return int(prefix[1:])
    except (IndexError, ValueError):
        return None


def _build_entries(
    qa_list: list[dict],
    dia_to_docs: dict[str, set[str]],
    session_docs: dict[int, set[str]],
) -> list[GroundTruthEntry]:
    """Build ground truth entries from QA annotations."""
    entries: list[GroundTruthEntry] = []

    for qa in qa_list:
        category_num = qa.get("category", 0)
        category = CATEGORY_NAMES.get(
            category_num, f"unknown-{category_num}"
        )

        # Resolve evidence dia_ids to doc_ids
        doc_ids: set[str] = set()
        for dia_id in qa.get("evidence", []):
            if dia_id in dia_to_docs:
                doc_ids.update(dia_to_docs[dia_id])
            else:
                # Fallback: map to all docs in that session
                session_num = _extract_session_number(dia_id)
                if (
                    session_num is not None
                    and session_num in session_docs
                ):
                    doc_ids.update(session_docs[session_num])

        # Get answer (adversarial uses adversarial_answer)
        answer = qa.get(
            "answer", qa.get("adversarial_answer")
        )

        if not doc_ids:
            continue

        entries.append(
            GroundTruthEntry(
                query=qa["question"],
                expected_doc_ids=tuple(sorted(doc_ids)),
                category=category,
                expected_answer=answer,
            )
        )

    return entries
