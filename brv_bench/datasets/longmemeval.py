"""LongMemEval dataset transformer.

Transforms LongMemEval JSON (from HuggingFace xiaowu0162/longmemeval-cleaned)
into brv-bench's canonical BenchmarkDataset format.

LongMemEval structure (ICLR 2025):
- 500 evaluation instances across 6 question types
- Each instance has a haystack of chat sessions (user/assistant turns)
- Evidence sessions identified by answer_session_ids
- Three variants: oracle (evidence only), S (~40 sessions), M (~500 sessions)

Corpus strategy:
- Each unique haystack session becomes one CorpusDocument
- Sessions are deduplicated across questions (many questions share sessions)
- doc_id = the session_id from the dataset (e.g., "answer_4be1b6b4_2")

Evidence mapping:
- answer_session_ids directly maps to doc_ids (no transformation needed)
"""

import json
from pathlib import Path

from brv_bench.datasets import register
from brv_bench.types import (
    BenchmarkDataset,
    CorpusDocument,
    GroundTruthEntry,
    PromptConfig,
)

CATEGORY_MAP: dict[str, str] = {
    "single-session-user": "single-session-user",
    "single-session-assistant": "single-session-assistant",
    "single-session-preference": "single-session-preference",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-update",
    "multi-session": "multi-session",
}


def transform(raw_path: Path) -> BenchmarkDataset:
    """Transform LongMemEval JSON into BenchmarkDataset.

    Args:
        raw_path: Path to a LongMemEval JSON file (oracle, S, or M).

    Returns:
        BenchmarkDataset with deduplicated session corpus and
        500 ground-truth entries.
    """
    with open(raw_path) as f:
        raw_data = json.load(f)

    corpus = _build_corpus(raw_data)
    entries = _build_entries(raw_data)

    return BenchmarkDataset(
        name="longmemeval",
        corpus=tuple(corpus),
        entries=tuple(entries),
    )


def _build_corpus(raw_data: list[dict]) -> list[CorpusDocument]:
    """Build deduplicated corpus from all haystack sessions.

    Each unique session (by session_id) becomes one CorpusDocument.
    Sessions are shared across questions, so we deduplicate globally.
    """
    seen: dict[str, CorpusDocument] = {}

    for entry in raw_data:
        session_ids = entry["haystack_session_ids"]
        sessions = entry["haystack_sessions"]
        dates = entry.get("haystack_dates", [])

        for idx, (sid, turns) in enumerate(zip(session_ids, sessions)):
            if sid in seen:
                continue

            date = dates[idx] if idx < len(dates) else ""
            content = _format_session(turns, date)

            seen[sid] = CorpusDocument(
                doc_id=sid,
                content=content,
                source=sid,
            )

    return sorted(seen.values(), key=lambda d: d.doc_id)


def _format_session(
    turns: list[dict],
    date: str,
) -> str:
    """Format a session's turns into readable text.

    Args:
        turns: List of {role, content, has_answer?} dicts.
        date: Session date string (e.g., "2023/04/10 (Mon) 17:50").

    Returns:
        Formatted session text.
    """
    lines: list[str] = []
    if date:
        lines.append(f"[{date}]")

    for turn in turns:
        role = turn["role"].capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _build_entries(raw_data: list[dict]) -> list[GroundTruthEntry]:
    """Build ground truth entries from evaluation instances."""
    entries: list[GroundTruthEntry] = []

    for item in raw_data:
        category = CATEGORY_MAP.get(
            item["question_type"],
            item["question_type"],
        )

        answer_session_ids = item.get("answer_session_ids", [])

        entries.append(
            GroundTruthEntry(
                query=item["question"],
                expected_doc_ids=tuple(sorted(answer_session_ids)),
                category=category,
                expected_answer=item.get("answer"),
            )
        )

    return entries


# =============================================================================
# Prompt templates for BrvCliAdapter
# =============================================================================

CURATE_TEMPLATE = """\
You are indexing a long-term chat assistant memory benchmark called \
LongMemEval into a context tree. Follow these rules EXACTLY. \
DO NOT READ ANY FILES in this directory. The only files you are \
allowed to read are from the context tree ./brv/context-tree/

## Context tree structure

- Domain = session topic cluster (group related sessions together)
- Topic  = session ID (e.g., `answer_4be1b6b4_2`)
- One file per session containing ONLY structured key facts (no transcript).

## What each context file MUST contain

1. **Metadata header** — session ID, date/time.
2. **Key facts** — Extract every factual statement, event, preference, \
opinion, request, recommendation, and personal detail mentioned by the \
user. Be exhaustive. Each fact on its own line.
3. **Curate limits** — ONE topic file for ONE session per curate task. \
DO NOT curate more than one topic file for each curate task.
4. **Context tree consistency** — Before creating a new topic file, \
investigate the existing context tree structure. Follow EXACTLY the \
same naming conventions, directory layout, and file format already \
established by previous curate tasks.

Do NOT store the raw transcript. Store only extracted facts.

## Curate tool usage

Use the UPSERT operation. Put ALL extracted content into \
`content.narrative.rules` as a single markdown string. Leave ALL other \
content fields empty — do NOT populate rawConcept, snippets, relations, \
narrative.structure, narrative.dependencies, narrative.diagrams, \
narrative.examples, or narrative.features.

Do NOT provide `domainContext`, `topicContext`, or `subtopicContext` — \
these generate extra context.md files that are not part of the benchmark \
corpus.

## Now index this content

```
doc_id: {doc_id}
source: {source}

{content}
```

Extract ALL facts. Do NOT summarize or skip any detail. \
Do NOT add information that is not in the transcript.
"""

QUERY_TEMPLATE = """\
Answer the following question using ONLY the chat history stored in \
the context tree. You MUST respond in EXACTLY this format — no extra text:

Follow these rules EXACTLY. DO NOT READ ANY FILES in this directory. \
The only files you are allowed to read are from the context tree \
./brv/context-tree/

ANSWER: <concise answer, few words only>
SOURCES: <comma-separated session IDs, e.g. answer_4be1b6b4_2>

Rules:
- Answer must be as concise as possible (names, dates, short phrases).
- SOURCES must list ONLY the session IDs whose key facts contain the evidence.
- SOURCES must NEVER be empty. Always list at least one session ID.
- If the information is not available, respond with:
    ANSWER: I don't have enough information to answer this question.
    SOURCES: none

## Now answer this question

- Question: {question}\
"""

PROMPT_CONFIG = PromptConfig(
    curate_template=CURATE_TEMPLATE,
    query_template=QUERY_TEMPLATE,
)

register("longmemeval", PROMPT_CONFIG)
