"Load a frozen code model and generate bounded, seeded program completions."

from __future__ import annotations

import logging

import torch

from .generation import Generation

LOGGER = logging.getLogger("roborepair.policies")


class RepairPolicy:
    """Frozen code model with seeded single-prompt and batch generation."""

    is_mock = False

    def __init__(self, model_cfg: dict, *, device: str = "cuda"):
        """Load the pinned model and tokenizer in bfloat16 on the requested device."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = model_cfg["name"]
        revision = model_cfg.get("revision")
        if revision in (None, "", "PIN_ME", "main"):
            raise ValueError("model.revision must identify an immutable checkpoint commit")
        quant = model_cfg.get("quantization", "none")
        if quant not in ("none", None):
            raise ValueError(f"Unsupported quantization: {quant!r}; use quantization: none")

        self.device = device
        self.model_name = name
        tokenizer_revision = model_cfg.get("tokenizer_revision", revision)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=tokenizer_revision)
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=torch.bfloat16
        ).to(device)
        model.requires_grad_(False)
        model.eval()

        self.model = model

    # utils
    def encode_chat_prompt(self, prompt: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def truncate_at_end_of_sequence(self, ids: list[int]) -> list[int]:
        eos = self.tokenizer.eos_token_id
        if eos in ids:
            return ids[: ids.index(eos) + 1]
        return ids

    # generate
    @torch.no_grad()
    def generate_batch(
        self,
        prompts,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 0,
        greedy: bool = False,
    ):
        """Wave-batched generation.

        ONE seed per wave, left-padded batch, deterministic for a fixed
        (prompts, seed) pair.
        """
        tok = self.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
            )
            for p in prompts
        ]
        enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(
            self.device
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        if greedy:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        seqs = self.model.generate(enc.input_ids, attention_mask=enc.attention_mask, **gen_kwargs)
        in_len = enc.input_ids.shape[1]
        out = []
        for i, row in enumerate(seqs):
            comp = self.truncate_at_end_of_sequence(row[in_len:].tolist())
            n_prompt = int(enc.attention_mask[i].sum())
            out.append(
                Generation(
                    text=tok.decode(comp, skip_special_tokens=True),
                    prompt_token_ids=enc.input_ids[i, in_len - n_prompt :].tolist()
                    if n_prompt
                    else [],
                    completion_token_ids=comp,
                    per_token_logprobs=None,
                    n_prompt_tokens=n_prompt,
                    n_completion_tokens=len(comp),
                )
            )
        return out

    @torch.no_grad()
    def generate(
        self,
        prompts,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 0,
        greedy: bool = False,
        num_return_sequences: int = 1,
    ):
        """Seeded generation. One call per prompt keeps per-episode seeds paired."""
        out = []
        for prompt in prompts:
            in_ids = self.encode_chat_prompt(prompt)
            input_ids = torch.tensor([in_ids], dtype=torch.long, device=self.device)
            attn = torch.ones_like(input_ids)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if greedy:
                gen_kwargs.update(do_sample=False)
            else:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            seqs = self.model.generate(input_ids, attention_mask=attn, **gen_kwargs)
            for row in seqs:
                comp = self.truncate_at_end_of_sequence(row[len(in_ids) :].tolist())
                text = self.tokenizer.decode(comp, skip_special_tokens=True)
                out.append(
                    Generation(
                        text=text,
                        prompt_token_ids=list(in_ids),
                        completion_token_ids=comp,
                        per_token_logprobs=None,
                        n_prompt_tokens=len(in_ids),
                        n_completion_tokens=len(comp),
                    )
                )
        return out
