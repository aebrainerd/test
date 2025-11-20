# Tweet Entropy Overlay (Chrome MV3 Extension)

Overlay per-token entropy and surprise metrics on Tweets (X) using any OpenAI-compatible API that returns logprobs with echo enabled.

## Features

- Scores only visible Tweets using `IntersectionObserver` and debouncing.
- Per-token spans colored by entropy (blue → red), tooltips list entropy (H), surprise (S), and top-k predictions.
- Lower-bound entropy badge (because only top-k mass is available).
- Per-Tweet collapse toggle plus mini legend.
- Session cache to avoid re-fetching the same Tweet while browsing.
- Options page for API base URL, key, model, and top-k.
- Graceful banners for missing API keys or API failures.

## Installation

1. `git clone` this repository and open `chrome://extensions` in Chrome.
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the repository folder.
4. Open the extension **Options** and enter your API base URL, API key, model, and desired top-k.

## API configuration

The extension expects an OpenAI-compatible `/v1/completions` endpoint that supports `logprobs` and `echo`.

Example command for a local vLLM server:

```
vllm serve <model_name> --port 8000 --api-key token --enable-logprobs
```

Then set `API Base URL` to `http://localhost:8000` and `Model` to your deployed model name.

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

- **No overlays and a banner appears:** Ensure the API key is set in Options.
- **API errors:** Check the base URL, model name, and that your endpoint supports `logprobs`+`echo`.
- **Slow overlays:** Reduce `top-k` or scroll slowly to give the model time to respond.
