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
        if not isinstance(value, list) or not key.startswith("session_"):
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
MANDATORY: emit exactly one `<bv-topic>` HTML document with \
path="{source}/{doc_id}". These values are fixed. Do NOT invent, \
rename, or replace them. Do NOT create any other topic files \
(no "indexing-guide" topic, no "guidelines" topic, no documentation \
topic). One curate call = one topic file at the path above.

You are indexing a long-term conversation dataset called LoCoMo into a \
context tree. Follow these rules EXACTLY. DO NOT READ ANY FILES in this \
directory. The only files you are allowed to read are from the context \
tree `./brv/context-tree/`.

## Context tree structure

- Domain = conversation ID (e.g. `conv_26`) — the source field above.
- Topic  = session number (e.g. `session_1`, `session_2`) — the doc_id field above.
- One file per session at `.brv/context-tree/<domain>/<topic>.html`.

Example file tree:

```
.brv/context-tree/
├── conv_26/
│   ├── session_1.html
│   ├── session_2.html
│   └── ...
├── conv_30/
│   ├── session_1.html
│   └── ...
└── ...
```

## Output contract

Emit a single `<bv-topic>` HTML document. The first character of your \
output must be `<` (the opening of `<bv-topic>`); the last characters \
must be `</bv-topic>`. Do NOT wrap in a code fence; do NOT add any \
prose preamble or trailing commentary. The document MUST contain:

1. **`<bv-topic>` attributes** — `path="{source}/{doc_id}"`, a \
keyword-rich `title` for retrieval, a one-line `summary`, and \
`tags`/`keywords` if available. The writer injects `createdat` and \
`updatedat` automatically — do NOT set them manually.

2. **Conversation metadata** — emit one `<bv-timestamp>` element with \
the session date/time, and one `<bv-author>` element listing the \
speakers (comma-separated).

3. **Key facts** — emit each factual statement, event, plan, \
preference, opinion, and personal detail as a SEPARATE `<bv-fact>` \
sibling element with attributes:
   - `subject="<key_concept_in_snake_case>"` (e.g. `support_group_attendance`)
   - `category="<personal|project|preference|convention|team|environment>"`
   - `value="<extracted_value>"` (the structured value of the fact)
   The element's text content is the canonical statement preserving \
the exact wording where possible. ONE fact per `<bv-fact>` element. \
Do NOT merge multiple facts.

Be exhaustive. Each statement gets its own `<bv-fact>`. Do NOT omit \
details. Do NOT store the raw transcript. Do NOT write narrative \
prose blocks — every fact is a typed `<bv-fact>` sibling.

Do NOT use any `<bv-rule>`, `<bv-task>`, or `<bv-decision>` elements \
for these conversational facts — those are reserved for actionable \
rules / tasks / decisions, which conversational data does not contain.

Do NOT mention or store JSON-schema field names like \
`narrative.rules`, `rawConcept.flow`, or `content.facts` — those \
belong to a deprecated curate-tool API. The output is HTML.

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

### Expected output (`.brv/context-tree/conv_26/session_1.html`):

```
<bv-topic path="conv_26/session_1" title="Caroline LGBTQ Support Group Counseling Career Melanie Lake Sunrise Painting" summary="Caroline shares experience at LGBTQ support group; plans counseling career. Melanie discusses lake sunrise painting and swimming with kids." tags="conversation,support,career,hobby" keywords="lgbtq,counseling,mental_health,painting,lake_sunrise,swimming">
<bv-timestamp>2023-05-08T13:56:00</bv-timestamp>
<bv-author>Caroline, Melanie</bv-author>
<bv-fact subject="support_group_attendance" category="personal" value="2023-05-07">Caroline attended a LGBTQ support group on 7 May 2023.</bv-fact>
<bv-fact subject="support_group_impact" category="personal" value="found_stories_inspiring">Caroline found the transgender stories inspiring.</bv-fact>
<bv-fact subject="emotional_outcome" category="personal" value="accepted_and_courageous">The support group made Caroline feel accepted and gave her courage.</bv-fact>
<bv-fact subject="career_intent" category="personal" value="counseling_mental_health">Caroline plans to continue her education and explore career options in counseling or mental health.</bv-fact>
<bv-fact subject="painting_subject" category="personal" value="lake_sunrise">Melanie painted a lake sunrise last year that is special to her.</bv-fact>
<bv-fact subject="family_activity" category="personal" value="swimming_with_kids">Melanie is going swimming with her kids.</bv-fact>
</bv-topic>
```

## Now index this content

```
doc_id: {doc_id}
source: {source}

{content}
```

Extract ALL facts. Do NOT summarize or skip any detail. \
Do NOT add information that is not in the transcript. \
Emit exactly ONE `<bv-topic>` document at the mandated path. \
The output starts with `<bv-topic` and ends with `</bv-topic>`.
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
