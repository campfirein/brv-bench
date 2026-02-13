"""Default judge prompt template.

Adapted from Hindsight's LoCoMo evaluation methodology — generous
binary correctness scoring with brief reasoning for auditability.
"""

DEFAULT_JUDGE_PROMPT = """\
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
counted as CORRECT.

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
