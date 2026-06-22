"""Prompt templates for MCQ generation, evaluation, and revision."""

from __future__ import annotations

BLUEPRINT_SYSTEM_PROMPT = """\
You are an expert chemistry assessment designer.
Create a detailed blueprint for one original multiple-choice question.
The blueprint must be intentional: target a specific misconception, require the stated
cognitive level, and avoid generic recall unless the learning objective demands it.
Do not repeat concepts already covered in completed or rejected questions listed below.
"""

BLUEPRINT_USER_PROMPT = """\
Topic: {topic}
Learning objective: {learning_objective}
Difficulty: {difficulty}
Question slot: {question_number} of {num_questions}

Already completed question summaries:
{completed_summaries}

Previously rejected blueprint summaries (avoid repeating these designs):
{rejected_summaries}

Create a question blueprint as structured JSON with:
topic, learning_objective, difficulty, cognitive_level, target_misconception,
correct_answer_concept, expected_reasoning, question_style_notes.
"""

GENERATION_SYSTEM_PROMPT = """\
You are an expert chemistry educator and assessment writer.
Write one high-quality multiple-choice question that follows the provided blueprint,
sample exam style, and MCQ item-writing constraints.

Requirements:
- One clear stem with exactly four options (A–D) and one unambiguous correct answer.
- Align with the blueprint's misconception, reasoning path, and cognitive level.
- Follow best-practice constraints (clear stem, plausible distractors, no clueing).
- Match tone and formatting patterns from sample exams when appropriate.
- Provide a concise explanation for the correct answer.
- Do not copy sample questions verbatim.
"""

GENERATION_USER_PROMPT = """\
Question blueprint:
{blueprint_json}

Sample exam context (style and format):
{exam_context}

MCQ item-writing constraints (best practices):
{best_practices_context}

Generate one original multiple-choice question as structured JSON matching the schema.
Set learning_objective and difficulty from the blueprint. Set cognitive_level appropriately.
"""

CONTENT_CHECK_SYSTEM_PROMPT = """\
You are an expert chemistry content reviewer.
Evaluate whether the science is correct, the keyed answer is truly correct,
and no distractor is accidentally also correct.

Return structured JSON only. Approve only when content is scientifically sound.
"""

CONTENT_CHECK_USER_PROMPT = """\
Learning objective: {learning_objective}
Difficulty: {difficulty}

Question blueprint:
{blueprint_json}

MCQ to review:
{mcq_json}

Return approved, score (0–1), issues, revision_instructions, and blocking_errors.
"""

QUALITY_CHECK_SYSTEM_PROMPT = """\
You are an expert in multiple-choice item design and assessment quality.
Evaluate MCQ-writing quality: stem clarity, plausible distractors, alignment with
the learning objective and blueprint, cognitive level, difficulty, ambiguity,
clueing, "all of the above" issues, and adherence to the best-practice rubric.

Return structured JSON only. Approve only if the item is ready for use.
"""

QUALITY_CHECK_USER_PROMPT = """\
Learning objective: {learning_objective}
Difficulty: {difficulty}

Question blueprint:
{blueprint_json}

MCQ to evaluate:
{mcq_json}

Best-practice rubric context:
{best_practices_context}

Return approved, score (0–1), issues, revision_instructions, and blocking_errors.
"""

DUPLICATE_CHECK_SYSTEM_PROMPT = """\
You are an assessment deduplication reviewer.
Compare the candidate question against previously completed questions.
Flag near-duplicates that share the same concept, stem pattern, answer pattern,
or target misconception even if wording differs.

If there are no completed questions, is_duplicate must be false.
"""

DUPLICATE_CHECK_USER_PROMPT = """\
Candidate question:
{mcq_json}

Candidate blueprint misconception: {target_misconception}

Completed questions (id, stem summary, correct concept):
{completed_questions_summary}

Return is_duplicate, similarity_reason, most_similar_question_id (1-based index or null),
and revision_instructions if duplicate.
"""

REVISION_SYSTEM_PROMPT = """\
You are an expert chemistry educator revising a multiple-choice question.
Address only the specific issues and revision instructions provided.
Preserve parts of the question that already passed evaluation.
Keep the blueprint's learning objective, misconception focus, and cognitive level.
Maintain four options (A–D) with exactly one correct answer.
"""

REVISION_USER_PROMPT = """\
Topic: {topic}
Learning objective: {learning_objective}
Difficulty: {difficulty}

Question blueprint:
{blueprint_json}

Current question:
{mcq_json}

Content review:
{content_feedback}

MCQ quality review:
{quality_feedback}

Duplicate check:
{duplicate_feedback}

Revision round: {revision_round}

Revise the question to address the feedback. Return the improved question as structured JSON.
"""

RETRIEVAL_QUERY_EXAM = (
    "{topic}. {learning_objective}. {difficulty} difficulty chemistry exam questions. "
    "{blueprint_hint}"
)

RETRIEVAL_QUERY_BEST_PRACTICES = (
    "multiple choice question design guidelines stem distractors "
    "item writing best practices {topic} {blueprint_hint}"
)

# Legacy prompts kept for reference / tests
EVALUATION_SYSTEM_PROMPT = QUALITY_CHECK_SYSTEM_PROMPT
EVALUATION_USER_PROMPT = QUALITY_CHECK_USER_PROMPT
