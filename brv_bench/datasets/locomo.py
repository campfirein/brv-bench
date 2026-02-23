"""LoCoMo dataset transformer.

Transforms LoCoMo's raw JSON (locomo10.json) into brv-bench's
canonical BenchmarkDataset format.

LoCoMo structure:
- 10 conversations, each with multiple sessions
- Each session has dialog turns with dia_ids (e.g., "D1:3")
- QA: questions with evidence pointing to dia_ids

Corpus strategy:
- Each session's full chat (all turns, both speakers) becomes
  one CorpusDocument.
- doc_id = session_N (matches topic folder in context tree)
- source = sample_id (conversation id, used as domain folder)

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
        sample_id = sample["sample_id"].replace("-", "_")
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

    doc_id = session_N (topic folder name in context tree).
    source = sample_id (domain folder name).

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
        doc_id = f"session_{session_num}"

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
                source=sample_id,
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
                doc_ids.add(f"session_{session_num}")

        # Get answer (adversarial uses adversarial_answer)
        answer = qa.get("answer", qa.get("adversarial_answer"))

        if not doc_ids:
            continue

        query = qa["question"]

        entries.append(
            GroundTruthEntry(
                query=query,
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

- Domain = conversation ID (e.g. `conv_26`)
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
doc_id: session_1
source: conv_26

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
# Session 1 — conv_26

**Source:** conv_26
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

QUERY_TEMPLATE = "{question}"

JUSTIFIER_TEMPLATE = """\
You are a helpful expert assistant answering questions from users \
based on the provided context.

# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct \
evidence in the memories
4. If the memories contain contradictory information or multiple \
instances of an event, say them all
5. Always convert relative time references to specific dates, months, \
or years.
6. Be as specific as possible when talking about people, places, and \
events
7. If the answer is not explicitly stated in the memories, use logical \
reasoning based on the information available to answer (e.g. calculate \
duration of an event from different memories).

Context:

{context}

Question: {question}
Answer:\
"""

PROMPT_CONFIG = PromptConfig(
    curate_template=CURATE_TEMPLATE,
    query_template=QUERY_TEMPLATE,
    justifier_template=JUSTIFIER_TEMPLATE,
)

register("locomo", PROMPT_CONFIG)
