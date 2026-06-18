"""Prompt templates for MCQ generation, evaluation, and revision."""

from __future__ import annotations

GENERATION_SYSTEM_PROMPT = """\
You are an expert chemistry educator and assessment writer.
Generate high-quality multiple-choice questions that match the style, structure,
and difficulty of the sample exam content provided.

Requirements:
- Write one clear stem with exactly four options (A–D) and one unambiguous correct answer.
- Align the question with the given learning objective, topic, and difficulty.
- Match the tone and formatting patterns seen in the sample exams when appropriate.
- Provide a concise explanation for the correct answer.
- Set cognitive_level to recall, application, analysis, or evaluation as appropriate.
- Do not copy sample questions verbatim; create original items in a similar style.
"""

GENERATION_USER_PROMPT = """\
Topic: {topic}
Learning objective: {learning_objective}
Difficulty: {difficulty}
Question number: {question_number} of {num_questions}

Sample exam context (use for style and content patterns):
{exam_context}

Generate one original multiple-choice question as structured JSON matching the schema.
"""

EVALUATION_SYSTEM_PROMPT = """\
You are an expert in multiple-choice item design and assessment quality.
Evaluate the question against established MCQ best practices.

Check for:
- unclear stem
- multiple correct answers
- weak distractors
- grammatical clues (e.g., "a/an", longest option)
- overly obvious correct answer
- negative wording ("NOT", "EXCEPT") unless justified
- inconsistent option length
- mismatch between learning objective and question content
- cognitive level too low when higher-order reasoning is requested

Use the best-practice rubric context below. Be specific and actionable.
"""

EVALUATION_USER_PROMPT = """\
Target learning objective: {learning_objective}
Requested difficulty: {difficulty}

MCQ to evaluate:
{mcq_json}

Best-practice rubric context:
{best_practices_context}

Return a structured evaluation with approved (true/false), feedback, issues, and suggestions.
Approve only if the question is ready for use with no major flaws.
"""

REVISION_SYSTEM_PROMPT = """\
You are an expert chemistry educator revising a multiple-choice question.
Preserve the original learning objective and topic focus while fixing evaluator feedback.
Keep the same general concept unless the feedback requires a substantive rewrite.
Maintain four options (A–D) with exactly one correct answer.
"""

REVISION_USER_PROMPT = """\
Topic: {topic}
Learning objective: {learning_objective}
Difficulty: {difficulty}

Current question:
{mcq_json}

Evaluator feedback:
{evaluator_feedback}

Issues identified:
{issues}

Revision round: {revision_round}

Revise the question to address the feedback. Return the improved question as structured JSON.
"""

RETRIEVAL_QUERY_EXAM = "{topic}. {learning_objective}. {difficulty} difficulty chemistry exam questions."

RETRIEVAL_QUERY_BEST_PRACTICES = (
    "multiple choice question design guidelines stem distractors "
    "item writing best practices {topic}"
)
