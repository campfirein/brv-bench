"""LoCoMo dataset transformer.

Transforms LoCoMo's raw JSON (locomo10.json) into brv-bench's
canonical BenchmarkDataset format.

LoCoMo structure:
- 10 conversations, each with multiple sessions
- Each session has dialog turns with dia_ids (e.g., "D1:3")
- QA: questions with evidence pointing to dia_ids

Corpus strategy:
- Each session's full chat (all turns, both speakers) becomes
  one CorpusDocument. doc_id format: {sample_id}_s{session_num}.

Evidence mapping:
- QA evidence dia_ids are mapped to session doc_ids by
  extracting the session number from the dia_id prefix.
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

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "commonsense",
    4: "single-hop",
    5: "adversarial",
}


def transform(raw_path: Path) -> BenchmarkDataset:
    """Transform LoCoMo JSON into BenchmarkDataset.

    Args:
        raw_path: Path to locomo10.json.

    Returns:
        BenchmarkDataset with session-chat corpus
        and QA-based entries.
    """
    with open(raw_path) as f:
        raw_data = json.load(f)

    all_corpus: list[CorpusDocument] = []
    all_entries: list[GroundTruthEntry] = []

    for sample in raw_data:
        sample_id = sample["sample_id"]
        corpus = _build_corpus(sample_id, sample["conversation"])
        all_corpus.extend(corpus)

        entries = _build_entries(sample_id, sample["qa"])
        all_entries.extend(entries)

    return BenchmarkDataset(
        name="locomo",
        corpus=tuple(all_corpus),
        entries=tuple(all_entries),
    )


def _build_corpus(
    sample_id: str,
    conversation: dict,
) -> list[CorpusDocument]:
    """Build corpus documents from session chat.

    Each session becomes one CorpusDocument containing
    the full dialog (all turns, both speakers).

    The conversation dict contains metadata keys
    (speaker_a, speaker_b, session_N_date_time) alongside
    session turn lists (session_N).

    Returns:
        List of CorpusDocuments, one per session.
    """
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    corpus: list[CorpusDocument] = []

    for key, value in conversation.items():
        if not isinstance(value, list):
            continue
        session_num = int(key.replace("session_", ""))
        doc_id = f"{sample_id}_s{session_num}"

        date_time = conversation.get(f"{key}_date_time", "")

        lines: list[str] = []
        if date_time:
            lines.append(f"[{date_time}]")
        lines.append(f"Conversation between {speaker_a} and {speaker_b}")
        for turn in value:
            speaker = turn["speaker"]
            text = turn.get("text", "")
            caption = turn.get("blip_caption", "")
            if caption:
                text = f"{text} [image: {caption}]"
            lines.append(f"{speaker}: {text}")

        corpus.append(
            CorpusDocument(
                doc_id=doc_id,
                content="\n".join(lines),
                source=key,
            )
        )

    return corpus


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
    sample_id: str,
    qa_list: list[dict],
) -> list[GroundTruthEntry]:
    """Build ground truth entries from QA annotations."""
    entries: list[GroundTruthEntry] = []

    for qa in qa_list:
        category_num = qa.get("category", 0)
        category = CATEGORY_NAMES.get(category_num, f"unknown-{category_num}")

        # Resolve evidence dia_ids to session doc_ids
        doc_ids: set[str] = set()
        for dia_id in qa.get("evidence", []):
            session_num = _extract_session_number(dia_id)
            if session_num is not None:
                doc_ids.add(f"{sample_id}_s{session_num}")

        # Get answer (adversarial uses adversarial_answer)
        answer = qa.get("answer", qa.get("adversarial_answer"))

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


# =============================================================================
# Prompt templates for BrvCliAdapter
# =============================================================================

CURATE_TEMPLATE = """\
You are indexing a long-term conversation dataset called LoCoMo into a \
context tree. Follow these rules EXACTLY. DO NOT READ ANY FILES in this directory. The only files you are allowed to read are from the context tree ./brv/context-tree/

## Context tree structure

- Domain = conversation ID (e.g. `conv-26`)
- Topic  = session number (e.g. `session-1`, `session-2`)
- One file per session containing ONLY structured key facts (no transcript).

Example file tree:

```
.brv/context-tree/
├── conv_26/
│   ├── session_1/
│   │   └── key_facts.md
│   ├── session_2/
│   │   └── key_facts.md
│   └── ...
├── conv_30/
│   ├── session_1/
│   │   └── key_facts.md
│   └── ...
└── ...
```

## What each context file MUST contain

1. **Metadata header** — doc_id, session source key, date/time, speakers.
2. **Key facts** — Extract every factual statement, event, speaker names, plan, preference, \
opinion, and personal detail mentioned. Be exhaustive. Each fact on its own \
line. Do NOT omit any detail.
3. **Curate limits** - ONE topic file for ONE session per curate task. DO NOT curate more than one topic file for each curate task.
4. **Context tree consistency** — Before creating a new topic file, investigate the existing context tree structure. Follow EXACTLY the same naming conventions, directory layout, and file format already established by previous curate tasks.

Do NOT store the raw transcript. Store only extracted facts.

## Curate tool usage

Use the UPSERT operation. Put ALL extracted content into `content.narrative.rules` \
as a single markdown string. Leave ALL other content fields empty — do NOT populate \
rawConcept, snippets, relations, narrative.structure, narrative.dependencies, \
narrative.diagrams, narrative.examples, or narrative.features.

Do NOT provide `domainContext`, `topicContext`, or `subtopicContext` — these generate \
extra context.md files that are not part of the benchmark corpus.

## Example

### Input (what you are given):

```
doc_id: conv-26_s1
source: session_1

[1:56 pm on 8 May, 2023]
Conversation between Caroline and Melanie
Caroline: I went to a LGBTQ support group yesterday and it was so powerful.
Melanie: What happened that was so awesome?
Caroline: The transgender stories were so inspiring!
Caroline: The support group has made me feel accepted and given me courage.
Caroline: Gonna continue my edu and check out career options.
Caroline: I'm keen on counseling or working in mental health.
Melanie: Yeah, I painted that lake sunrise last year! It's special to me.
Melanie: I'm off to go swimming with the kids.
```

### Expected output (.brv/context-tree/conv_26/session_1/key_facts.md):

```markdown
# Session 1 — conv-26_s1

**Source:** session_1
**Date/Time:** 1:56 pm on 8 May, 2023
**Speakers:** Caroline, Melanie

## Key Facts

- Caroline went to a LGBTQ support group the day before (7 May 2023)
- Caroline found the transgender stories inspiring
- The support group made Caroline feel accepted and gave her courage
- Caroline plans to continue her education and explore career options
- Caroline is interested in counseling or mental health work
- Melanie painted a lake sunrise last year; it is special to her
- Melanie is going swimming with her kids
```

## Now index this content

```
doc_id: {doc_id}
source: {source}

{content}
```

Extract ALL facts. Do NOT summarize or skip any detail. \
Do NOT add information that is not in the transcript. \
Follow EXACTLY the file structure as the example above. \
The key concepts MUST NOT be too short or vague.
"""

QUERY_TEMPLATE = """\
Answer the following question using ONLY the conversation context stored in \
the context tree. You MUST respond in EXACTLY this format — no extra text:

Follow these rules EXACTLY. DO NOT READ ANY FILES in this directory. The only files you are allowed to read are from the context tree ./brv/context-tree/

ANSWER: <concise answer, few words only>
SOURCES: <comma-separated doc_ids, e.g. conv-26_s1, conv-26_s3>

Rules:
- Answer must be as concise as possible (names, dates, short phrases).
- SOURCES must list ONLY the doc_ids whose key facts contain the evidence.
- SOURCES must NEVER be empty. Always list at least one doc_id.

## Examples

- Question: When did Caroline go to the LGBTQ support group?
- Your answer:
    ANSWER: 7 May 2023
    SOURCES: conv-26_s1

- Question: What career path has Caroline decided to pursue?
- Your answer:
    ANSWER: counseling or mental health for transgender people
    SOURCES: conv-26_s1, conv-26_s4

- Question: Where has Melanie camped?
- Your answer:
    ANSWER: beach, mountains, forest
    SOURCES: conv-26_s4, conv-26_s6, conv-26_s8

- Note: Again, just two two-part answer: ANSWER and SOURCES. No summary or further discussion.
## Now answer this question

- Question: {question}\
"""

PROMPT_CONFIG = PromptConfig(
    curate_template=CURATE_TEMPLATE,
    query_template=QUERY_TEMPLATE,
)

register("locomo", PROMPT_CONFIG)
