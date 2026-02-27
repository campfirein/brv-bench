"""Judge prompt templates.

Adapted from Hindsight's LoCoMo evaluation methodology and LongMemEval's
category-specific evaluation criteria.  Generous binary correctness scoring
with brief reasoning for auditability.

Category-specific prompts (LongMemEval):
- single-session-user / single-session-assistant / multi-session:
    Standard correctness with equivalence and intermediate-step tolerance.
- temporal-reasoning:
    Off-by-one tolerance for day/week/month counts.
- knowledge-update:
    Accepts updated answers alongside stale information.
- single-session-preference:
    Evaluates personalised recall, not exact phrasing.

Default prompt (LoCoMo-style):
    Generous semantic matching with time-format tolerance.
"""

# ── Common suffix ────────────────────────────────────────────────────
# Appended to every category preamble to supply the Q/A pair and
# enforce the JSON output format expected by ``parse_verdict()``.

_JUDGE_SUFFIX = """\

Question: {question}
Gold answer: {expected_answer}
Generated answer: {predicted_answer}
First, provide a short (one sentence) explanation of your reasoning. \
Short reasoning is preferred.

Respond with EXACTLY this JSON format, nothing else:
{{"reasoning": "<one sentence>", "verdict": "correct"}}
or
{{"reasoning": "<one sentence>", "verdict": "incorrect"}}
"""

_STANDARD_PREAMBLE = """\
I will give you a question, a correct answer, and a response from a model. \
Please evaluate if the model response contains the correct answer.

Evaluation criteria:
- Set verdict to "correct" if the response contains the correct answer.
- Set verdict to "correct" if the response is equivalent to the correct \
answer or contains all the intermediate steps to get the correct answer.
- Set verdict to "incorrect" if the response is incorrect, missing key \
information, or only contains a subset of the information required by \
the answer.\
"""

_TEMPORAL_REASONING_PREAMBLE = """\
I will give you a question, a correct answer, and a response from a model. \
Please evaluate if the model response contains the correct answer.

Evaluation criteria:
- Set verdict to "correct" if the response contains the correct answer.
- Set verdict to "correct" if the response is equivalent to the correct \
answer or contains all the intermediate steps to get the correct answer.
- Set verdict to "incorrect" if the response is incorrect, missing key \
information, or only contains a subset of the information required by \
the answer.
- Do NOT penalise off-by-one errors for counts of days, weeks, or months. \
If the question asks for the number of days/weeks/months and the model \
makes an off-by-one error (e.g., predicting 19 days when the answer is \
18), the model's response should still be considered correct.\
"""

_KNOWLEDGE_UPDATE_PREAMBLE = """\
I will give you a question, a correct answer, and a response from a model. \
Please evaluate if the model response contains the correct answer.

Evaluation criteria:
- Set verdict to "correct" if the response contains the correct answer.
- If the response contains some previous information along with an \
updated answer, the response should be considered correct as long as \
the updated answer matches the required answer.
- Set verdict to "incorrect" if the response does not contain the \
correct (updated) answer.\
"""

_PREFERENCE_PREAMBLE = """\
I will give you a question, a desired personalised response, and a \
response from a model. Please evaluate if the model response satisfies \
the desired response.

Evaluation criteria:
- Set verdict to "correct" if the response satisfies the desired \
response.
- The model does NOT need to reflect all the points in the desired \
response. The response is correct as long as it recalls and utilises \
the user's personal information correctly.
- Set verdict to "incorrect" if the response fails to recall or \
utilise the user's personal information.\
"""

_DEFAULT_PREAMBLE = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. \
You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should \
know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that \
includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous \
with your grading - as long as it touches on the same topic as the \
gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, \
month, year, etc. The generated answer might be much longer or use \
relative time references (like "last Tuesday" or "next month"), but \
you should be generous with your grading - as long as it refers to \
the same date or time period as the gold answer, it should be counted \
as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), \
consider it CORRECT if it's the same date.
There's an edge case where the actual answer can't be found in the \
data and in that case the gold answer will say so (e.g. 'You did not \
mention this information.'); if the generated answer says that it \
cannot be answered or it doesn't know all the details, it should be \
counted as CORRECT.\
"""

_CATEGORY_PREAMBLES: dict[str, str] = {
    "single-session-user": _STANDARD_PREAMBLE,
    "single-session-assistant": _STANDARD_PREAMBLE,
    "multi-session": _STANDARD_PREAMBLE,
    "temporal-reasoning": _TEMPORAL_REASONING_PREAMBLE,
    "knowledge-update": _KNOWLEDGE_UPDATE_PREAMBLE,
    "single-session-preference": _PREFERENCE_PREAMBLE,
}

DEFAULT_JUDGE_PROMPT = _DEFAULT_PREAMBLE + _JUDGE_SUFFIX


def get_judge_prompt(category: str) -> str:
    """Return the judge prompt template for *category*.

    Falls back to the default LoCoMo-style prompt when the category
    has no specialised preamble.
    """
    preamble = _CATEGORY_PREAMBLES.get(category)
    if preamble is None:
        return DEFAULT_JUDGE_PROMPT
    return preamble + _JUDGE_SUFFIX
