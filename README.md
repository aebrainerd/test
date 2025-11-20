# Tweet Entropy Overlay (Chrome MV3 Extension)

Overlay per-token entropy and surprise metrics on Tweets (X) using the bundled local Qwen backend that returns logprobs with echo enabled.

## Features

- Scores only visible Tweets using `IntersectionObserver` and debouncing.
- Per-token spans colored by entropy (blue → red), tooltips list entropy (H), surprise (S), and top-k predictions.
- Lower-bound entropy badge (because only top-k mass is available).
- Per-Tweet collapse toggle plus mini legend.
- Session cache to avoid re-fetching the same Tweet while browsing.
- Options page for model and top-k (defaults to the local Qwen backend at `http://localhost:8000`).
- Graceful banners for local API failures.

## Installation

1. `git clone` this repository and open `chrome://extensions` in Chrome.
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the repository folder.
4. Start the local backend (see below); open the extension **Options** and set the model (defaults to `Qwen/Qwen1.5-4B`) and desired top-k.

## API configuration

The extension always calls the bundled backend at `http://localhost:8000/v1/completions` and does not require an API key. You can change the model name from the Options page if you run a different local checkpoint.

## Built-in local backend (Qwen ~4B)

This repository also ships a minimal OpenAI-compatible scoring backend that runs a Qwen base model locally and exposes `/v1/completions`.

Requirements:

- Python 3.10+
- Enough RAM/GPU memory for the selected model (default: `Qwen/Qwen1.5-4B`, roughly 8–10 GB in fp16; CPU fp32 works but is slower)

Install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Start the server (defaults to `Qwen/Qwen1.5-4B`):

```
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

Environment variables:

- `QWEN_MODEL`: override the model name (e.g., `Qwen/Qwen1.5-1.8B` to reduce memory).
- `QWEN_DEVICE_MAP`: passed to `transformers` `device_map` (default `auto`).

API behavior:

- Only supports `max_tokens=0` and `echo=true` requests to score prompt tokens.
- Returns `choices[0].logprobs.tokens`, `token_logprobs`, and `top_logprobs` in OpenAI-compatible format.

The request payload is:

```json
{
  "model": "<MODEL>",
  "prompt": "<TWEET TEXT>",
  "max_tokens": 0,
  "temperature": 0,
  "logprobs": <TOP_K>,
  "echo": true
}
```

Responses should include `choices[0].logprobs.tokens`, `token_logprobs`, and `top_logprobs`.

## Known limitations

- Entropy is a lower bound when only top-k probabilities are available.
- Token alignment is best-effort using greedy substring matching; minor misalignments can occur with complex tokenization or multi-codepoint emoji.
- The extension skips DM pages and editable composers but may not detect every protected surface.

## Troubleshooting

- **No overlays and a banner appears:** Verify the local backend is running on `http://localhost:8000`.
- **API errors:** Confirm the backend process is healthy and the selected model name matches what the server loaded.
- **Slow overlays:** Reduce `top-k` or scroll slowly to give the model time to respond.
