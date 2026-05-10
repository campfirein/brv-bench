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
        category = item["question_type"]

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
MANDATORY: emit exactly one `<bv-topic>` HTML document with \
path="{source}/{doc_id}". These values are fixed. Do NOT invent, \
rename, or replace them. Do NOT create any other topic files \
(no "indexing-guide" topic, no "guidelines" topic). One curate call = \
one topic file at the path above.

You are indexing a long-term chat assistant memory benchmark called \
LongMemEval into a context tree. Follow these rules EXACTLY. \
DO NOT READ ANY FILES in this directory. The only files you are \
allowed to read are from the context tree `./brv/context-tree/`.

## Context tree structure

- Domain = question ID (e.g., `gpt4_2655b836`) — the source field above.
- Topic  = session label (e.g., `session_1`, `session_2`) — the doc_id field above.
- One file per session at `.brv/context-tree/<domain>/<topic>.html`.

Example file tree:

```
.brv/context-tree/
├── gpt4_2655b836/
│   ├── session_1.html
│   ├── session_2.html
│   └── session_3.html
├── gpt4_2487a7cb/
│   ├── session_1.html
│   └── session_2.html
└── ...
```

## Output contract

Emit a single `<bv-topic>` HTML document. The first character of your \
output must be `<` (the opening of `<bv-topic>`); the last characters \
must be `</bv-topic>`. Do NOT wrap in a code fence; do NOT add any \
prose preamble or trailing commentary. The document MUST contain:

1. **`<bv-topic>` attributes:**
   - `path="{source}/{doc_id}"` — REQUIRED, exact value
   - `title` — REQUIRED keyword-rich BM25-friendly title (8–14 words):
     * Noun phrase only — NOT a full sentence, no trailing period
     * Always include proper nouns: people names, pet names, place \
names, brand names, product names, specific values/counts
     * Do NOT start with "User" or "Assistant"
     * Do NOT use vague phrases like "Various Topics", "Personal \
Conversation", or "Discussion About"
     * Match strategy to content type:
       - Personal facts (pet name/breed, name change, certification, \
purchase): include ALL named entities and specifics
       - Temporal events (trip, ceremony, party): include EVENT NAME, \
VENUE, and named people
       - Tracked metrics (bike count, follower count, score): include \
exact number and all item names
       - Per-session facts in a running tally (vet cost, plant bought): \
include the specific item, amount, and named subject (e.g. pet name)
       - Preference/recommendation sessions: include the user's topic \
and any named products/places recommended
   - `summary` — RECOMMENDED one-line semantic summary
   - `tags`/`keywords` — comma-separated retrieval tags. The writer \
injects `createdat`/`updatedat` automatically.

2. **Session metadata** — emit one `<bv-timestamp>` element with the \
session date/time.

3. **Key facts** — emit each factual statement, event, preference, \
opinion, request, recommendation, and personal detail (from BOTH \
user AND assistant) as a SEPARATE `<bv-fact>` sibling element with:
   - `subject="<key_concept_in_snake_case>"`
   - `category="<personal|project|preference|convention|team|environment>"`
   - `value="<extracted_value>"`
   The element's text content is the canonical statement preserving \
the exact wording where possible. ONE fact per `<bv-fact>` element. \
Do NOT merge multiple facts.

Be exhaustive. Each statement gets its own `<bv-fact>`. Do NOT omit \
details. Do NOT store the raw transcript. Do NOT write narrative \
prose blocks — every fact is a typed `<bv-fact>` sibling.

Do NOT use `<bv-rule>`, `<bv-task>`, or `<bv-decision>` elements for \
conversational facts — those are reserved for actionable rules / \
tasks / decisions.

Do NOT mention or store JSON-schema field names like \
`narrative.rules`, `rawConcept.flow`, or `content.facts` — those \
belong to a deprecated curate-tool API. The output is HTML.

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

### Expected output (`.brv/context-tree/gpt4_2655b836/session_2.html`):

```
<bv-topic path="gpt4_2655b836/session_2" title="Car First Service GPS Not Working Brake Pads Confirmed Fine Reset Suggested" summary="User reports GPS broken after first car service; assistant suggests reset. Mechanic confirmed brake pads fine." tags="car,service,gps,maintenance" keywords="gps,car_service,brake_pads,reset">
<bv-timestamp>2023-04-10T14:47:00</bv-timestamp>
<bv-fact subject="car_service_status" category="personal" value="first_service_completed">User just got car back from its first service.</bv-fact>
<bv-fact subject="gps_status" category="personal" value="not_working">GPS system is not working after the service.</bv-fact>
<bv-fact subject="assistant_recommendation" category="preference" value="reset_gps">Assistant suggested resetting the GPS system.</bv-fact>
<bv-fact subject="user_planned_action" category="personal" value="will_try_reset">User will try resetting the GPS.</bv-fact>
<bv-fact subject="brake_pads_status" category="personal" value="confirmed_fine">Mechanic confirmed brake pads are fine.</bv-fact>
</bv-topic>
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
Emit exactly ONE `<bv-topic>` document at the mandated path. \
The output starts with `<bv-topic` and ends with `</bv-topic>`.\
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
