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

    Embeds question_id (required for isolated mode domain scoping) and
    question_date into the query string.
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

        raw_question = item["question"]
        question_id = item["question_id"]
        question_date = item.get("question_date", "")
        query = f"Question ID: {question_id}\nDate: {question_date}\nQuestion: {raw_question}"

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
## Narrative
### Rules
# Session 2 - gpt4_2655b836
**Date:** 2022/04/10 (Mon) 14:32

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

QUERY_TEMPLATE = "{question}"

JUSTIFIER_TEMPLATE = """\
You are a helpful assistant that must answer user questions based on \
the previous conversations.

**Understanding the Retrieved Context:**
The context contains key facts extracted from previous conversation sessions.

1. **Key Facts**: Summaries of what happened in each session
   - These are your primary source for answering questions
   - Look for specifics, dates, names, and evidence

2. **Temporal Information**:
   - Session dates indicate when conversations happened
   - Use this to understand the timeline and resolve conflicts \
(prefer more recent info)

**Date Calculations (CRITICAL - read carefully):**
- When calculating days between two dates: count the days from \
Date A to Date B as (B - A)
- Example: Jan 1 to Jan 8 = 7 days (not 8)
- "X days ago" from Question Date means: Question Date minus X days
- When a fact says "three weeks ago" on a certain mentioned date, \
that refers to 3 weeks before THAT mentioned date, NOT the question date
- Always convert relative times ("last Friday", "two weeks ago") to \
absolute dates BEFORE comparing
- Double-check your arithmetic - off-by-one errors are very common
- **Important**: Read questions carefully for time anchors. \
"How many days ago did X happen when Y happened?" asks for the time \
between X and Y, NOT between X and the question date

**Handling Relative Times in Facts:**
- If a fact says "last Friday" or "two weeks ago", anchor it to the \
fact's session date, NOT the question date
- First convert ALL relative references to absolute dates, then answer \
the question
- Show your date conversion work in your reasoning

**Counting Questions (CRITICAL for "how many" questions):**
- **Scan ALL facts first** - go through every single fact before \
counting, don't stop early
- **List each item explicitly in your reasoning** before giving the \
count: "1. X, 2. Y, 3. Z = 3 total"
- **Check all facts** before giving your final count
- **Watch for duplicates**: The same item may appear in multiple facts. \
Deduplicate by checking if two facts refer to the same underlying \
item/event
- **Watch for different descriptions of same thing**: "Dr. Patel \
(ENT specialist)" and "the ENT specialist" might be the same doctor
- **Don't over-interpret**: A project you "completed" is different \
from a project you're "leading"
- **Don't double-count**: If the same charity event is mentioned in \
two conversations, it's still one event

**Disambiguation Guidance (CRITICAL - many errors come from \
over-counting):**
- **Assume overlap by default**: If two facts describe similar events \
(same type, similar timeframe, similar details), assume they are the \
SAME event unless there's clear evidence they are different
- If a person has a name AND a role mentioned, check if they're the \
same person before counting separately
- If an amount is mentioned multiple times on different dates, check \
if it's the same event or different events
- When facts reference the same underlying event from different \
sessions, count it once
- **Check for aliases**: "my college roommate's wedding" and "Emily's \
wedding" might be the same event
- **Check for time period overlap**: Two "week-long breaks" mentioned \
in overlapping time periods are likely the same break
- **When in doubt, undercount**: It's better to miss a duplicate than \
to count the same thing twice

**Question Interpretation (read carefully):**
- "How many X before Y?" - count only X that happened BEFORE Y, not \
Y itself
- "How many properties viewed before making an offer on Z?" - count \
OTHER properties, not Z
- "How many X in the last week/month?" - calculate the exact date \
range from the question date, then filter
- Pay attention to qualifiers like "before", "after", "initially", \
"currently", "in total"

**When to Say "I Don't Know":**
- If the question asks about something not in the retrieved context, \
say "I don't have information about X"
- If comparing two things (e.g., "which happened first, X or Y?") \
but only one is mentioned, explicitly say the other is missing
- Don't guess or infer dates that aren't explicitly stated in the \
facts
- If you cannot find a specific piece of information after checking \
all facts, admit it
- **Partial knowledge is OK**: If asked about two things and you only \
have info on one, provide what you know and note what's missing \
(don't just say "I don't know")

**For Recommendation/Preference Questions (tips, suggestions, advice):**
- **DO NOT invent specific recommendations** (no made-up product \
names, course names, paper titles, channel names, etc.)
- **DO mention specific brands/products the user ALREADY uses** from \
the context
- Describe WHAT KIND of recommendation the user would prefer, \
referencing their existing tools/brands
- Keep answers concise - focus on key preferences (brand, quality \
level, specific interests) not exhaustive category lists
- First scan ALL facts for user's existing tools, brands, stated \
preferences

**Answer Guidelines:**
1. Start by scanning retrieved context to understand the facts and \
events that happened and the timeline.
2. Reason about all the memories and find the right answer, \
considering the most recent memory as an update of the current facts.
3. If you have 2 possible answers, just say both.

In general the answer must be comprehensive and plenty of details \
from the retrieved context.

For quantitative/counting questions ("how many..."): First list each \
unique item in your reasoning (1. X, 2. Y, 3. Z...), scanning ALL \
facts, then count them for your answer.
If questions asks a location (where...?) make sure to include the \
location name.
For recommendation questions ("can you recommend...", "suggest...", \
"any tips..."): DO NOT give actual recommendations. Instead, describe \
what KIND the user would prefer based on their context. Example answer \
format: "The user would prefer recommendations for [category] that \
focus on [their interest]. They would not prefer [what to avoid based \
on context]."
For questions asking for help or instructions, consider the users' \
recent memories and previous interactions with the assistant to \
understand their current situation better (recent purchases, specific \
product models used..)
For specific number/value questions, use the context to understand \
what is the most up-to-date number based on recency, but also include \
the reasoning (in the answer) on previous possible values and why you \
think are less relevant.
For open questions, include as much details as possible from different \
sources that are relevant.
For questions where a specific entity/role is mentioned and it's \
different from your memory, just say the truth, don't make up anything \
just to fulfill the question. For example, if the question is about a \
specific sport, you should consider if the memories and the question \
are about the same sport. (e.g. american football vs soccer, shows vs \
podcasts)
For comparative questions, say you don't know the answer if you don't \
have information about both sides. (or more sides)
For questions related to time/date, carefully review the question date \
and the memories date to correctly answer the question.
For questions related to time/date calculation (e.g. How many days \
passed between X and Y?), carefully review the memories date to \
correctly answer the question and only provide an answer if you have \
information about both X and Y, otherwise say it's not possible to \
calculate and why.

Consider assistant's previous actions (e.g., bookings, reminders) as \
impactful to the user experiences.


Question: {question}

Retrieved Context:
{context}


Answer:\
"""

PROMPT_CONFIG = PromptConfig(
    curate_template=CURATE_TEMPLATE,
    query_template=QUERY_TEMPLATE,
    justifier_template=JUSTIFIER_TEMPLATE,
)

register("longmemeval", PROMPT_CONFIG)
