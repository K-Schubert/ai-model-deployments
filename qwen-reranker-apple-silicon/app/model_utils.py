"""
Utility functions and model / tokenizer singletons.

This rewrites the original vLLM implementation to use
Transformers (AutoModelForCausalLM).  It works entirely on CPU,
so it is compatible with an arm64 Docker image running on an M-series Mac.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# Model & tokenizer (loaded once when the module is imported)
# ---------------------------------------------------------------------

_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME).eval()

# Place the model on GPU if available; otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------------------------------------------------
# Fixed tokens / template pieces
# ---------------------------------------------------------------------

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

max_length_default: int = 8192  # OpenAI “ranger” default

# Chat-style wrapper (identical to the original system / assistant prompt)
prefix = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
suffix = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# ---------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------


def format_instruction(instruction: str | None, query: str, doc: str) -> str:
    """Return a plain-text prompt fragment for one (query, document) pair."""
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


def process_inputs(
    pairs: list[str],
    *,
    max_length: int = max_length_default,
):
    """
    Tokenise the batch of prompt strings and move everything to the
    same device as the model.

    The final layout for each example is:

        prefix_tokens + <tokenised pair> + suffix_tokens
    """
    # First pass: truncate the **content** so we never exceed the budget
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )

    # Add the fixed chat wrapper
    for i, ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens

    # Pad to a single tensor and move to model.device
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for k in inputs:
        inputs[k] = inputs[k].to(model.device)

    return inputs


@torch.no_grad()
def compute_logits(inputs) -> list[float]:
    """
    Forward pass and convert the last-token logits to probabilities.

    The output is a list of probabilities (one per input) that the
    answer is **“yes”** (True).  Equivalent to the old vLLM score.
    """
    # logits shape: (batch, sequence, vocab)
    logits = model(**inputs).logits[:, -1, :]

    # Collect the logits for "no" and "yes"
    false_vec = logits[:, token_false_id]
    true_vec = logits[:, token_true_id]

    # Shape → (batch, 2)   [0] => “no”,  [1] => “yes”
    two_logit = torch.stack([false_vec, true_vec], dim=1)

    # Softmax over the two classes, take exp to get probabilities
    probs = torch.nn.functional.log_softmax(two_logit, dim=1).exp()

    # Return the probability for class 1 (“yes”)
    return probs[:, 1].tolist()
