"""Generation records shared by model and mock policies."""

from dataclasses import dataclass


@dataclass
class Generation:
    """Generated program text, token IDs, optional log probabilities, and token counts."""

    text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    per_token_logprobs: object | None
    n_prompt_tokens: int
    n_completion_tokens: int
