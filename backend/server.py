import os
import time
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen1.5-4B")
DEFAULT_DEVICE_MAP = os.environ.get("QWEN_DEVICE_MAP", "auto")


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=DEFAULT_DEVICE_MAP,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


class CompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    prompt: str
    max_tokens: int = 0
    temperature: float = 0.0
    logprobs: int = 5
    echo: bool = True

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v != 0:
            raise ValueError("This server only supports scoring with max_tokens=0")
        return v

    @validator("echo")
    def validate_echo(cls, v):
        if not v:
            raise ValueError("This server requires echo=true to return prompt tokens")
        return v


class TokenLogprobs(BaseModel):
    tokens: List[str]
    token_logprobs: List[Optional[float]]
    top_logprobs: List[Optional[Dict[str, float]]]
    text_offset: Optional[List[int]] = None


class Choice(BaseModel):
    text: str
    index: int
    logprobs: TokenLogprobs
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]


app = FastAPI(title="Qwen OpenAI-Compatible Scoring API")
tokenizer, model = load_model_and_tokenizer(DEFAULT_MODEL)


def build_response(prompt: str, model_name: str, top_k: int) -> CompletionResponse:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)

    input_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    token_logprobs: List[Optional[float]] = []
    top_logprobs: List[Optional[Dict[str, float]]] = []

    for i, token_id in enumerate(input_ids):
        current_logprobs = logprobs[i]
        token_logprob = current_logprobs[token_id].item()
        token_logprobs.append(token_logprob)

        top_values, top_indices = torch.topk(current_logprobs, k=min(top_k, current_logprobs.shape[-1]))
        top_dict = {
            tokenizer.convert_ids_to_tokens(idx.item()): val.item()
            for val, idx in zip(top_values, top_indices)
        }
        top_logprobs.append(top_dict)

    usage = {
        "prompt_tokens": len(tokens),
        "completion_tokens": 0,
        "total_tokens": len(tokens),
    }

    response = CompletionResponse(
        id=f"cmpl-{int(time.time() * 1000)}",
        object="text_completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            Choice(
                text=prompt,
                index=0,
                logprobs=TokenLogprobs(
                    tokens=tokens,
                    token_logprobs=token_logprobs,
                    top_logprobs=top_logprobs,
                    text_offset=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
    return response


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    model_name = request.model or DEFAULT_MODEL
    if model_name != tokenizer.name_or_path:
        try:
            global tokenizer, model
            tokenizer, model = load_model_and_tokenizer(model_name)
        except Exception as exc:  # pragma: no cover - startup errors
            raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    try:
        response = build_response(request.prompt, model_name, request.logprobs)
        return response
    except Exception as exc:  # pragma: no cover - runtime errors
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok", "model": tokenizer.name_or_path}
