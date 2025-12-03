from omegaconf import DictConfig


def get_task_name(cfg: DictConfig):
    dataset = getattr(cfg, "dataset", "")
    subset = getattr(cfg, "subset", "")
    # Treat empty or 'default' subset as no subset component
    if subset in (None, "", "default"):
        return dataset
    return f"{dataset}-{subset}"


def user_prompt_glue_mrpc(chunk):
    """Format MRPC examples into few-shot user prompt."""
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        s1 = ex["sentence1"]
        s2 = ex["sentence2"]
        label = ex.get("label", None)
        label_text = "true" if label == 1 else "false" if label == 0 else "UNKNOWN"
        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Sentence 1: {s1}\n"
            f"Sentence 2: {s2}\n"
            f"Equivalence: {label_text}\n"
        )
    return "\n".join(user_parts).strip()


def user_prompt_super_glue_cb(chunk):
    """Format SuperGLUE CB examples into few-shot user prompt.

    Expected HF fields: "premise", "hypothesis", "label" (0/1/2 or None).
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        label = ex.get("label", None)

        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Label: {label}\n"
        )

    return "\n".join(user_parts).strip()


def user_prompt_super_glue_rte(chunk):
    """Format SuperGLUE RTE examples into few-shot user prompt.

    Expected HF fields: "premise", "hypothesis", "label" (binary: 0/1).
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        premise = ex["premise"]
        hypothesis = ex["hypothesis"]
        label = ex.get("label", None)

        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Label: {label}\n"
        )

    return "\n".join(user_parts).strip()


def user_prompt_super_glue_copa(chunk):
    """Format SuperGLUE COPA examples into few-shot user prompt.

    Expected HF fields: "premise", "choice1", "choice2", "question" ("cause"|"effect"), "label" ("choice1", "choice2").
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        premise = ex["premise"]
        choice1 = ex["choice1"]
        choice2 = ex["choice2"]
        question = ex["question"]
        label = ex.get("label", None)

        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Premise: {premise}\n"
            f"Choice1: {choice1}\n"
            f"Choice2: {choice2}\n"
            f"Question: {question}\n"
            f"Label: {label}\n"
        )

    return "\n".join(user_parts).strip()


def user_prompt_super_glue_wsc(chunk):
    """Format SuperGLUE WSC examples into few-shot user prompt.

    Expected HF fields: "text", "span1_text" (pronoun), "span2_text" (candidate), "label" (0/1).

    [features]
    {'idx': Value('int32'),
    'label': ClassLabel(names=['False', 'True']),
    'span1_index': Value('int32'),
    'span1_text': Value('string'),
    'span2_index': Value('int32'),
    'span2_text': Value('string'),
    'text': Value('string')}
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        text = ex["text"]
        span1 = ex["span1_text"]
        span2 = ex["span2_text"]
        label = ex.get("label", None)

        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Text: {text}\n"
            f"Span1Text: {span1}\n"
            f"Span2Text: {span2}\n"
            f"Label: {label}\n"
        )

    return "\n".join(user_parts).strip()


def user_prompt_super_glue_wic(chunk):
    """Format SuperGLUE WiC examples into few-shot user prompt.

    Expected HF fields: "sentence1", "sentence2", "word", "label" (0/1 or bool).

    [features]
    {'end1': Value('int32'),
    'end2': Value('int32'),
    'idx': Value('int32'),
    'label': ClassLabel(names=['False', 'True']),
    'sentence1': Value('string'),
    'sentence2': Value('string'),
    'start1': Value('int32'),
    'start2': Value('int32'),
    'word': Value('string')}
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        s1 = ex["sentence1"]
        s2 = ex["sentence2"]
        end1 = ex["end1"]
        end2 = ex["end2"]
        start1 = ex["start1"]
        start2 = ex["start2"]
        word = ex["word"]
        label = ex.get("label", None)
        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Word: {word}\n"
            f"Sentence1: {s1}\n"
            f"Start1: {start1}\n"
            f"End1: {end1}\n"
            f"Sentence2: {s2}\n"            
            f"Start2: {start2}\n"
            f"End2: {end2}\n"
            f"Label: {label}\n"
        )
    return "\n".join(user_parts).strip()


def user_prompt_super_glue_boolq(chunk):
    """Format SuperGLUE BoolQ examples into few-shot user prompt.

    Expected HF fields: "question", "passage", "label" (0/1 or bool).

    [features]
    {'idx': Value('int32'),
    'label': ClassLabel(names=['False', 'True']),
    'passage': Value('string'),
    'question': Value('string')}
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        q = ex["question"]
        p = ex["passage"]
        label = ex.get("label", None)
        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Passage: {p}\n"
            f"Question: {q}\n"
            f"Label: {label}\n"
        )
    return "\n".join(user_parts).strip()


def user_prompt_super_glue_multirc(chunk):
    """Format SuperGLUE MultiRC examples into few-shot user prompt.

    Expected HF fields: "passage" (or "paragraph"), "question", "answers" (list of {"text", "label"}).
    [features]
    {'answer': Value('string'),
    'idx': {'answer': Value('int32'),
            'paragraph': Value('int32'),
            'question': Value('int32')},
    'label': ClassLabel(names=['False', 'True']),
    'paragraph': Value('string'),
    'question': Value('string')}
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        paragraph = ex.get("passage") or ex.get("paragraph") # paragraph i guess?
        question = ex.get("question")
        answer = ex.get("answers", [])
        label = ex.get("label")
        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Paragraph: {paragraph}\n"
            f"Question: {question}\n"
            f"Candidate Answer: {answer}\n"
            f"Label: {label}\n"
        )
    return "\n".join(user_parts).strip()


def user_prompt_super_glue_record(chunk):
    """Format SuperGLUE ReCoRD examples into few-shot user prompt.

    Expected HF fields: "passage", "query", "entities", "answers".
    [features]
    {'answers': List(Value('string')),
    'entities': List(Value('string')),
    'entity_spans': {'end': List(Value('int32')),
                    'start': List(Value('int32')),
                    'text': List(Value('string'))},
    'idx': {'passage': Value('int32'), 'query': Value('int32')},
    'passage': Value('string'),
    'query': Value('string')}
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        passage = ex["passage"]
        query = ex["query"]
        entitiy_spans = ex("entity_spans", {})
        entities = ex.get("entities", [])
        answers = ex.get("answers", [])

        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Passage: {passage}\n"
            f"Query: {query}\n"
            f"Entity Spans: {entitiy_spans}\n"
            f"Entities: {entities}\n"
            f"Answers: {answers}\n"
        )
    return "\n".join(user_parts).strip()


def user_prompt_legalbench_legal_reasoning_causality(chunk):
    """Format LegalBench legal_reasoning_causality examples into few-shot user prompt.

    Expected HF fields: "text", "answer" ("Yes"/"No").
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        text = ex["text"]
        answer = ex["answer"]

        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Text: {text}\n"
            f"Answer: {answer}\n"
        )
    return "\n".join(user_parts).strip()


PROMPT_BUILDERS = {
    # GLUE
    "glue-mrpc": user_prompt_glue_mrpc,

    # SuperGLUE
    "super_glue-cb": user_prompt_super_glue_cb,
    "super_glue-rte": user_prompt_super_glue_rte,
    "super_glue-copa": user_prompt_super_glue_copa,
    "super_glue-wsc": user_prompt_super_glue_wsc,
    "super_glue-wic": user_prompt_super_glue_wic,
    "super_glue-boolq": user_prompt_super_glue_boolq,
    "super_glue-multirc": user_prompt_super_glue_multirc,
    "super_glue-record": user_prompt_super_glue_record,

    # LegalBench
    "legalbench-legal_reasoning_causality": user_prompt_legalbench_legal_reasoning_causality,

    # BIOSSES
    "biosses": None,  # placeholder; will be set below
}


def get_user_prompt_fn(cfg: DictConfig):
    # subset > dataset
    task = get_task_name(cfg)
    if task not in PROMPT_BUILDERS:
        raise ValueError(f"No user prompt builder registered for task '{task}'")
    return PROMPT_BUILDERS[task]

# Add BIOSSES builder now that function is defined above
def user_prompt_biosses(chunk):
    """Format BIOSSES examples into few-shot user prompt.

    Expected HF fields: "sentence1", "sentence2", "score" (float 0..5).
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        s1 = ex["sentence1"]
        s2 = ex["sentence2"]
        score = ex.get("score", None)
        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Sentence 1: {s1}\n"
            f"Sentence 2: {s2}\n"
            f"Score: {score}\n"
        )
    return "\n".join(user_parts).strip()

# Register the BIOSSES prompt builder
PROMPT_BUILDERS["biosses"] = user_prompt_biosses
