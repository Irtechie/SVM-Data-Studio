"""LLM prompt templates for the benchmark pipeline.

All prompts are optimised for instruction-tuned models in the 30B parameter
class (Qwen2.5-32B, Llama-3.3-70B, Mistral-Small-3.1, etc.) with large
context windows (≥128k tokens).

Design principles:
- Short, direct instructions (smaller models lose focus in long preambles)
- Inline JSON example so the model knows the exact schema from context
- "Output ONLY valid JSON" always repeated at the end
- Curly braces that are NOT placeholders are doubled: {{ }}
"""

# ── Labeling prompts ──────────────────────────────────────────────────────

LABELING_PROMPT_TABULAR = """\
You are labeling one example from the {dataset_name} dataset for a machine learning experiment.

Dataset description: {dataset_description}

Possible class labels (choose EXACTLY one): {class_names}

Example features:
{feature_dict}

Classify this example. Respond with ONLY valid JSON matching this schema:
{{"label": "<one of the class labels above>", "confidence": <float 0.0-1.0>, "reasoning": "<one sentence>"}}

Output ONLY the JSON object. No markdown, no explanation outside the JSON."""

LABELING_PROMPT_TEXT = """\
You are labeling a text example from the {dataset_name} dataset for a machine learning experiment.

Task: {task_description}

Possible class labels (choose EXACTLY one): {class_names}

Text to classify:
\"\"\"
{text}
\"\"\"

Classify this text. Respond with ONLY valid JSON matching this schema:
{{"label": "<one of the class labels above>", "confidence": <float 0.0-1.0>, "reasoning": "<one sentence>"}}

Output ONLY the JSON object. No markdown, no explanation outside the JSON."""

LABELING_PROMPT_UNCERTAINTY_RETRY = """\
A downstream classifier is uncertain about this example from the {dataset_name} dataset.
Re-examine it carefully before answering.

Possible class labels (choose EXACTLY one): {class_names}

Example:
{example_data}

Your previous label: {previous_label}
Your previous confidence: {previous_confidence}

Consider edge cases and ambiguity. Respond with ONLY valid JSON matching this schema:
{{"label": "<one of the class labels above>", "confidence": <float 0.0-1.0>, "reasoning": "<two or three sentences, especially if you changed your answer>", "changed": <true or false>}}

Output ONLY the JSON object. No markdown, no explanation outside the JSON."""

# ── Report generation prompt ──────────────────────────────────────────────

REPORT_GENERATION_PROMPT = """\
Write a clear educational report about the following machine learning experiment.
Your audience is someone learning data analytics. Use plain language, be specific to the numbers, and do NOT invent details.

EXPERIMENT METADATA:
- Dataset: {dataset_name}
- Description: {dataset_description}
- Data type: {data_type}
- Examples: {n_examples}
- Classes ({n_classes}): {class_names}
- LLM used for labeling: {llm_model}
- Optional techniques enabled: {optional_techniques}

CORE RESULTS:
- LLM label agreement with ground truth: {llm_agreement_pct}%
- SVM on LLM labels — test accuracy: {svm_llm_accuracy}%
- SVM on true labels (control) — test accuracy: {svm_control_accuracy}%
- Performance gap (labeling cost): {gap}%
- CV variance — LLM-trained SVM: {variance_llm}
- CV variance — control SVM: {variance_control}

CONFUSION ANALYSIS:
- Most common LLM mislabel: {most_common_error}
- Class with highest LLM error rate: {worst_class}
- Class with lowest LLM error rate: {best_class}

OPTIONAL TECHNIQUE RESULTS:
{optional_results_block}

Write the report with these five sections:

**1. What we tested**
One paragraph explaining the experiment in plain language.

**2. What the LLM did**
How well the LLM labeled the data and where it struggled.

**3. What the SVM showed**
What the downstream classifier reveals about label quality.

**4. Why this happened**
Was this task well-suited to LLM labeling? Why or why not?

**5. What you should take away**
Practical guidance: would you use LLM labels for this task in production?

Be specific to the actual numbers above. Write at the level of a clear technical blog post."""

OPTIONAL_TECHNIQUE_EXPLANATION_PROMPT = """\
Explain the following optional analysis section for a machine learning experiment report.

Technique: {technique_name}
Dataset context: {dataset_description}

Technique results:
{technique_results}

Write 2–3 paragraphs in a clear educational voice. Be specific to the numbers.
Explain what the technique does, why it was applied to this dataset, and what the results mean for the user."""
