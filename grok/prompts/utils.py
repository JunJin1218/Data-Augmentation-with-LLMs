from omegaconf import DictConfig

def get_task_name(cfg: DictConfig):
    dataset = getattr(cfg, "dataset", "")
    subset = getattr(cfg, "subset", "")
    task = f"{dataset}-{subset}"
    return task

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
    """Format SuperGLUE CB examples into few-shot user prompt."""
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


PROMPT_BUILDERS = {
    "glue-mrpc": user_prompt_glue_mrpc,
    "super_glue-cb": user_prompt_super_glue_cb,
    # Additions for other SuperGLUE tasks registered below after definitions
    # "sst2": user_prompt_sst2,
}

def get_user_prompt_fn(cfg):
    # subset > dataset
    task = get_task_name(cfg)
    if task not in PROMPT_BUILDERS:
        raise ValueError(f"No user prompt builder registered for task '{task}'")
    return PROMPT_BUILDERS[task]

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

# Register RTE builder
PROMPT_BUILDERS["super_glue-rte"] = user_prompt_super_glue_rte


def user_prompt_super_glue_copa(chunk):
    """Format SuperGLUE COPA examples into few-shot user prompt.

    Expected HF fields: "premise", "choice1", "choice2", "question" ("cause"|"effect"), "label" (0/1).
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


# Register COPA and WSC builders
PROMPT_BUILDERS["super_glue-copa"] = user_prompt_super_glue_copa
PROMPT_BUILDERS["super_glue-wsc"] = user_prompt_super_glue_wsc


def user_prompt_super_glue_wic(chunk):
    """Format SuperGLUE WiC examples into few-shot user prompt.

    Expected HF fields: "sentence1", "sentence2", "word", "label" (0/1 or bool).
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        s1 = ex["sentence1"]
        s2 = ex["sentence2"]
        word = ex["word"]
        label = ex.get("label", None)
        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Word: {word}\n"
            f"Sentence1: {s1}\n"
            f"Sentence2: {s2}\n"
            f"Label: {label}\n"
        )
    return "\n".join(user_parts).strip()


def user_prompt_super_glue_boolq(chunk):
    """Format SuperGLUE BoolQ examples into few-shot user prompt.

    Expected HF fields: "question", "passage", "label" (0/1 or bool).
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


# Register WiC and BoolQ builders
PROMPT_BUILDERS["super_glue-wic"] = user_prompt_super_glue_wic
PROMPT_BUILDERS["super_glue-boolq"] = user_prompt_super_glue_boolq


def user_prompt_super_glue_multirc(chunk):
    """Format SuperGLUE MultiRC examples into few-shot user prompt.

    Expected HF fields: "passage" (or "paragraph"), "question", "answers" (list of {"text", "label"}).
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        passage = ex.get("passage") or ex.get("paragraph")
        question = ex.get("question")
        answers = ex.get("answers", [])
        user_parts.append(f"Example {idx_in_chunk}:\nPassage: {passage}\nQuestion: {question}")
        for ai, ans in enumerate(answers, start=1):
            a_text = ans.get("text")
            a_label = ans.get("label")
            user_parts.append(f"Answer {ai}: {a_text} | Label: {a_label}")
        user_parts.append("")
    return "\n".join(user_parts).strip()


# Register MultiRC builder
PROMPT_BUILDERS["super_glue-multirc"] = user_prompt_super_glue_multirc


def user_prompt_super_glue_record(chunk):
    """Format SuperGLUE ReCoRD examples into few-shot user prompt.

    Expected HF fields: "passage", "query", "entities", "answers".
    """
    user_parts = []
    for idx_in_chunk, ex in enumerate(chunk, start=1):
        passage = ex["passage"]
        query = ex["query"]
        entities = ex["entities"]
        answers = ex["answers"]
        
        user_parts.append(
            f"Example {idx_in_chunk}:\n"
            f"Passage: {passage}\n"
            f"Query: {query}\n"
            f"Entities: {entities}\n"
            f"Answers: {answers}\n"
        )
    return "\n".join(user_parts).strip()


# Register ReCoRD builder
PROMPT_BUILDERS["super_glue-record"] = user_prompt_super_glue_record
