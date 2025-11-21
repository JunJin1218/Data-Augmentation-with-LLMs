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
    "super_glue-cb": user_prompt_super_glue_cb
    # "rte": user_prompt_rte,
    # "sst2": user_prompt_sst2,
}

def get_user_prompt_fn(cfg):
    # subset > dataset
    task = get_task_name(cfg)
    if task not in PROMPT_BUILDERS:
        raise ValueError(f"No user prompt builder registered for task '{task}'")
    return PROMPT_BUILDERS[task]
