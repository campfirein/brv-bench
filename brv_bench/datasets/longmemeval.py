"""LongMemEval dataset transformer.

Transforms LongMemEval JSON (from HuggingFace xiaowu0162/longmemeval-cleaned)
into brv-bench's canonical BenchmarkDataset format.

LongMemEval structure (ICLR 2025):
- 500 evaluation instances across 6 question types
- Each instance has a haystack of chat sessions (user/assistant turns)
- Evidence sessions identified by answer_session_ids
- Three variants: oracle (evidence only), S (~40 sessions), M (~500 sessions)

Corpus strategy (per-question isolation):
- Each question's haystack sessions become separate CorpusDocuments
- doc_id  = session_N (1-based index in the haystack array)
- source  = question_id (used as domain folder in context tree)
- Sessions shared across questions are duplicated per question
- The has_answer field is stripped from messages (confidential)

Evidence mapping:
- answer_session_ids are mapped to session_N labels via their
  position in haystack_session_ids.
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
        BenchmarkDataset with per-question corpus and
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
    """Build per-question corpus from haystack sessions.

    Each question gets its own set of sessions. Sessions shared
    across questions are duplicated (one copy per question domain).
    doc_id = session_N (1-based), source = question_id.
    """
    corpus: list[CorpusDocument] = []
    seen: set[tuple[str, str]] = set()  # (question_id, session_label)

    for entry in raw_data:
        question_id = entry["question_id"]
        sessions = entry["haystack_sessions"]
        dates = entry.get("haystack_dates", [])

        for idx, turns in enumerate(sessions):
            session_label = f"session_{idx + 1}"
            key = (question_id, session_label)
            if key in seen:
                continue
            seen.add(key)

            date = dates[idx] if idx < len(dates) else ""
            content = _format_session(turns, date)

            corpus.append(
                CorpusDocument(
                    doc_id=session_label,
                    content=content,
                    source=question_id,
                )
            )

    return sorted(corpus, key=lambda d: (d.source, d.doc_id))


def _format_session(
    turns: list[dict],
    date: str,
) -> str:
    """Format a session's turns into readable text.

    Strips has_answer (confidential). Includes only role + content.
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
    """Build ground truth entries from evaluation instances.

    Embeds question_id and question_date into the query string
    so the query template can scope search to the right domain.
    Maps answer_session_ids to session_N labels.
    """
    entries: list[GroundTruthEntry] = []

    for item in raw_data:
        category = CATEGORY_MAP.get(
            item["question_type"],
            item["question_type"],
        )

        # Build lookup: original session_id → session_N label
        sid_to_label = {
            sid: f"session_{idx + 1}"
            for idx, sid in enumerate(item["haystack_session_ids"])
        }

        # Map answer_session_ids to session_N labels
        answer_labels = [
            sid_to_label[sid]
            for sid in item.get("answer_session_ids", [])
            if sid in sid_to_label
        ]

        # Embed question_id and question_date into query string
        question_id = item["question_id"]
        question_date = item.get("question_date", "")
        raw_question = item["question"]

        query = (
            f"Question ID: {question_id}\n"
            f"Date: {question_date}\n"
            f"Question: {raw_question}"
        )

        entries.append(
            GroundTruthEntry(
                query=query,
                expected_doc_ids=tuple(sorted(answer_labels)),
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

- Domain = question ID (e.g., `gpt4_2655b836`)
- Topic  = session label (e.g., `session_1`, `session_2`)
- One file per session containing ONLY structured key facts (no transcript).

Example file tree:

```
.brv/context-tree/
├── gpt4_2655b836/
│   ├── session_1/
│   │   └── key_facts.md
│   ├── session_2/
│   │   └── key_facts.md
│   └── session_3/
│       └── key_facts.md
├── gpt4_2487a7cb/
│   ├── session_1/
│   │   └── key_facts.md
│   └── session_2/
│       └── key_facts.md
└── ...
```

## What each context file MUST contain

1. **Metadata header** — session label (session_1, session_2, etc.), \
date/time of the chat session.
2. **Key facts** — Extract every factual statement, event, preference, \
opinion, request, recommendation, and personal detail mentioned by \
BOTH the user AND the assistant. Be exhaustive. Each fact on its own line. \
Do NOT omit any detail.
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

## Example

### Input (what you are given):

```
doc_id: session_2
source: gpt4_2655b836

[2023/04/10 (Mon) 14:47]
User: I just got my car back from its first service and the GPS isn't working.
Assistant: That sounds frustrating! Have you tried resetting the GPS?
User: Good idea, I'll try that. The mechanic also said the brake pads are fine.
Assistant: Great to hear about the brakes. Let me know if the reset fixes it.
```

### Expected output (.brv/context-tree/gpt4_2655b836/session_2/key_facts.md):

```markdown
# Session 2
**Date:** 2023/04/10 (Mon) 14:47

## Key Facts
- User just got car back from its first service
- GPS system is not working after the service
- Assistant suggested resetting the GPS system
- User will try resetting the GPS
- Mechanic confirmed brake pads are fine
```

## Now index this content

```
doc_id: {doc_id}
source: {source}

{content}
```

Extract ALL facts from BOTH user and assistant messages. \
Do NOT summarize or skip any detail. \
Do NOT add information that is not in the transcript. \
Follow EXACTLY the file structure shown in the example above. \
The key facts MUST NOT be too short or vague.\
"""

QUERY_TEMPLATE = """\
Answer the following question using ONLY the chat history stored in \
the context tree. You MUST respond in EXACTLY this format — no extra text:

ANSWER: <concise answer, few words only>
SOURCES: <comma-separated session labels, e.g. session_1, session_3>

Rules:
- The question below includes a Question ID. Search ONLY within the \
domain folder matching that Question ID in the context tree \
(e.g., ./brv/context-tree/<question_id>/).
- DO NOT read files outside of that domain folder. Reading other domains is COMPLETELY forbidden. \
- Answer must be as concise as possible (names, dates, short phrases).
- SOURCES must list ONLY the session labels (session_1, session_2, etc.) \
whose key facts contain the evidence.
- If the information is NOT available or the question contains a false \
premise, respond with:
    ANSWER: I don't have enough information to answer this question.
    SOURCES: none

## Examples

- Question ID: gpt4_2655b836
  Date: 2023/04/10 (Mon) 23:07
  Question: What was the first issue with my car after its first service?
- Your answer:
    ANSWER: GPS system not functioning correctly
    SOURCES: session_1, session_2, session_3

- Question ID: 66f24dbb
  Date: 2023/05/26 (Fri) 11:54
  Question: What did I buy for my sister's birthday gift?
- Your answer:
    ANSWER: a yellow dress
    SOURCES: session_1

- Question ID: gpt4_70e84552_abs
  Date: 2023/06/15 (Thu) 10:30
  Question: Which task did I complete first, fixing the fence or purchasing cows?
- Your answer:
    ANSWER: I don't have enough information to answer this question.
    SOURCES: none

- Note: Just the two-part answer: ANSWER and SOURCES. No summary or further discussion.

## Now answer this question

{question}\
"""

PROMPT_CONFIG = PromptConfig(
    curate_template=CURATE_TEMPLATE,
    query_template=QUERY_TEMPLATE,
)

register("longmemeval", PROMPT_CONFIG)
