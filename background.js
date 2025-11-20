const DEFAULT_SETTINGS = {
  apiBase: 'https://api.openai.com',
  apiKey: '',
  model: 'gpt-3.5-turbo-instruct',
  top_k: 5
};

async function getSettings() {
  const stored = await chrome.storage.local.get(['apiBase', 'apiKey', 'model', 'top_k']);
  return {
    apiBase: stored.apiBase || DEFAULT_SETTINGS.apiBase,
    apiKey: stored.apiKey || DEFAULT_SETTINGS.apiKey,
    model: stored.model || DEFAULT_SETTINGS.model,
    top_k: typeof stored.top_k === 'number' ? stored.top_k : DEFAULT_SETTINGS.top_k
  };
}

async function getFromSession(key) {
  const res = await chrome.storage.session.get(key);
  return res[key];
}

async function setInSession(key, value) {
  await chrome.storage.session.set({ [key]: value });
}

async function callModel(apiBase, apiKey, model, prompt, topK) {
  const body = {
    model,
    prompt,
    max_tokens: 0,
    temperature: 0,
    logprobs: topK,
    echo: true
  };

  const url = `${apiBase.replace(/\/$/, '')}/v1/completions`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify(body)
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`API error ${resp.status}: ${txt}`);
  }

  const data = await resp.json();
  return data;
}

function normalizeResponse(data) {
  const choice = data?.choices?.[0];
  if (!choice || !choice.logprobs) {
    throw new Error('No logprobs in response');
  }
  const tokens = choice.logprobs.tokens || [];
  const topLogprobs = choice.logprobs.top_logprobs || [];
  const per_pos = tokens.map((_, idx) => {
    const dist = topLogprobs[idx + 1];
    if (!dist) return null;
    const topk = Object.entries(dist).map(([token, logp]) => ({ token, logp }));
    return { topk };
  });
  return { tokens, per_pos };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === 'score_tweet') {
    handleScoreRequest(message)
      .then((result) => sendResponse(result))
      .catch((err) => sendResponse({ error: 'api_error', message: err.message }));
    return true;
  }
  return false;
});

async function handleScoreRequest(message) {
  const settings = await getSettings();
  if (!settings.apiKey) {
    return { error: 'missing_api_key' };
  }

  const sessionKey = `${message.cacheKey || ''}|${settings.model}|${settings.top_k}`;
  const cached = await getFromSession(sessionKey);
  if (cached) {
    return { ok: true, data: cached };
  }

  const data = await callModel(settings.apiBase, settings.apiKey, settings.model, message.text, settings.top_k);
  const normalized = normalizeResponse(data);
  await setInSession(sessionKey, normalized);
  return { ok: true, data: normalized };
}
