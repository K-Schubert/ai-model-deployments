from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

from model_utils import (
    format_instruction,
    process_inputs,
    compute_logits,
    max_length_default,
)

app = FastAPI()


class Pair(BaseModel):
    query: str
    document: str


class InferenceRequest(BaseModel):
    # A list of (query, document) pairs to be judged
    pairs: List[Pair]
    # Optional task description.  If omitted, a sensible default is used.
    instruction: str | None = None
    # Upper bound on context length (tokens) – keep 8 k by default
    max_length: int = max_length_default


@app.post("/infer")
async def infer(req: InferenceRequest):
    # Build the plain-text prompts for each (query, document) pair
    formatted_pairs = [
        format_instruction(req.instruction, p.query, p.document) for p in req.pairs
    ]

    # Tokenise + pad so everything is on the model’s device
    inputs = process_inputs(formatted_pairs, max_length=req.max_length)

    # Forward pass → probability that the correct answer is **“yes”**
    scores = compute_logits(inputs)

    return {"scores": scores}
