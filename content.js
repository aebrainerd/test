(function () {
  const DEFAULT_SETTINGS = {
    top_k: 5,
    model: 'Qwen/Qwen1.5-4B'
  };

  let settings = { ...DEFAULT_SETTINGS };
  let pending = new Set();
  let debounceTimer = null;
  const processed = new WeakSet();
  const observer = new IntersectionObserver(handleIntersect, { threshold: 0.2 });
  const mutationObserver = new MutationObserver(scanTweets);

  function log(...args) {
    console.debug('[TweetEntropy]', ...args);
  }

  function hashString(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = (h << 5) - h + str.charCodeAt(i);
      h |= 0;
    }
    return `h${Math.abs(h)}`;
  }

  async function initSettings() {
    const stored = await chrome.storage.local.get(['top_k', 'model']);
    settings = {
      ...settings,
      ...stored,
      top_k: typeof stored.top_k === 'number' ? stored.top_k : DEFAULT_SETTINGS.top_k,
      model: stored.model || DEFAULT_SETTINGS.model
    };
  }

  function scanTweets() {
    const articles = document.querySelectorAll('article[data-testid="tweet"]');
    articles.forEach((article) => {
      if (processed.has(article)) return;
      if (shouldSkipArticle(article)) return;
      processed.add(article);
      observer.observe(article);
    });
  }

  function shouldSkipArticle(article) {
    if (!article || article.dataset.teObserved) return true;
    const isDM = window.location.pathname.includes('/messages');
    if (isDM) return true;
    const editor = article.querySelector('[contenteditable="true"]');
    if (editor) return true;
    article.dataset.teObserved = '1';
    return false;
  }

  function handleIntersect(entries) {
    entries.forEach((entry) => {
      if (entry.isIntersecting && entry.intersectionRatio >= 0.2) {
        queueTweet(entry.target);
      }
    });
  }

  function queueTweet(article) {
    pending.add(article);
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(processQueue, 300);
  }

  function processQueue() {
    const queue = Array.from(pending);
    pending.clear();
    queue.forEach(scoreArticle);
  }

  function extractTweetText(article) {
    const textNodes = article.querySelectorAll('[data-testid="tweetText"]');
    if (!textNodes.length) return '';
    const parts = [];
    textNodes.forEach((node) => {
      node.childNodes.forEach((child) => {
        if (child.nodeType === Node.TEXT_NODE) {
          parts.push(child.textContent || '');
        } else if (child.textContent) {
          parts.push(child.textContent);
        }
      });
    });
    return parts.join('');
  }

  async function scoreArticle(article) {
    const tweetText = extractTweetText(article).trim();
    if (!tweetText) return;

    const container = ensureOverlayContainer(article);
    if (!container) return;
    if (container.dataset.loaded === '1') return;

    const cacheKey = hashString(`${tweetText}|${settings.model}|${settings.top_k}`);
    const body = container.querySelector('.te-body');
    if (body) body.textContent = 'Scoring tweet...';

    let response;
    try {
      response = await chrome.runtime.sendMessage({
        type: 'score_tweet',
        text: tweetText,
        cacheKey
      });
    } catch (err) {
      renderError(container, 'api_error');
      return;
    }

    if (!response || response.error || !response.ok) {
      renderError(container, response?.error);
      return;
    }

    renderOverlay(container, tweetText, response.data);
    container.dataset.loaded = '1';
  }

  function ensureOverlayContainer(article) {
    const target = article.querySelector('[data-testid="tweetText"]');
    if (!target) return null;
    let existing = article.querySelector('.te-overlay');
    if (existing) return existing;
    const overlay = document.createElement('div');
    overlay.className = 'te-overlay';

    const header = document.createElement('div');
    header.className = 'te-header';
    const title = document.createElement('span');
    title.textContent = 'Tweet Entropy Overlay';
    const toggle = document.createElement('button');
    toggle.textContent = 'Toggle';
    toggle.className = 'te-toggle';
    toggle.addEventListener('click', () => {
      overlay.classList.toggle('te-collapsed');
    });
    header.appendChild(title);
    header.appendChild(toggle);
    overlay.appendChild(header);

    const banner = document.createElement('div');
    banner.className = 'te-banner';
    banner.style.display = 'none';
    overlay.appendChild(banner);

    const legend = document.createElement('div');
    legend.className = 'te-legend';
    legend.innerHTML = '<span>Low</span><span class="te-gradient"></span><span>High</span>';
    overlay.appendChild(legend);

    const body = document.createElement('div');
    body.className = 'te-body';
    overlay.appendChild(body);

    target.parentElement?.appendChild(overlay);
    return overlay;
  }

  function renderError(container, code) {
    const banner = container.querySelector('.te-banner');
    if (!banner) return;
    banner.style.display = 'block';
    banner.textContent = 'Unable to score tweet (local API unavailable).';
    const body = container.querySelector('.te-body');
    if (body) body.textContent = '';
  }

  function renderOverlay(container, text, data) {
    const body = container.querySelector('.te-body');
    const banner = container.querySelector('.te-banner');
    if (banner) banner.style.display = 'none';
    if (!body) return;
    body.textContent = '';

    const alignments = alignTokensToText(data.tokens || [], text);
    const lowerBound = Boolean(settings.top_k && settings.top_k > 0);

    alignments.forEach((align, idx) => {
      const span = document.createElement('span');
      span.className = 'te-token';
      const tokenText = text.slice(align.start, align.end) || align.token || '';
      span.textContent = tokenText;

      const dist = data.per_pos?.[idx];
      if (dist && dist.topk && dist.topk.length) {
        const norm = normalizeTopk(dist.topk);
        const entropy = computeEntropy(norm);
        const surprise = computeSurprise(norm, data.tokens[idx]);
        span.style.background = entropyColor(entropy);
        span.dataset.tooltip = buildTooltip(entropy, surprise, norm);
      }

      body.appendChild(span);
    });

    if (lowerBound) {
      const pill = document.createElement('div');
      pill.className = 'te-pill';
      pill.textContent = 'Lower-bound entropy (top-k only)';
      body.appendChild(pill);
    }
  }

  function alignTokensToText(tokens, text) {
    const positions = [];
    let cursor = 0;
    tokens.forEach((tok) => {
      let idx = text.indexOf(tok, cursor);
      if (idx === -1) {
        idx = cursor;
      }
      if (idx > text.length) idx = text.length;
      if (idx > cursor) {
        positions.push({ start: cursor, end: idx, token: text.slice(cursor, idx) });
      }
      const end = Math.min(idx + tok.length, text.length);
      positions.push({ start: idx, end: end > idx ? end : idx, token: tok });
      cursor = end;
    });
    if (cursor < text.length) {
      positions.push({ start: cursor, end: text.length, token: text.slice(cursor) });
    }
    return positions;
  }

  function normalizeTopk(topk) {
    const probs = topk.map((entry) => Math.exp(entry.logp));
    const total = probs.reduce((a, b) => a + b, 0) || 1;
    return topk.map((entry, i) => ({
      token: entry.token,
      logp: entry.logp,
      prob: probs[i] / total
    }));
  }

  function computeEntropy(normTopk) {
    let h = 0;
    normTopk.forEach((entry) => {
      const p = entry.prob;
      if (p > 0) {
        h -= p * Math.log2(p);
      }
    });
    return h;
  }

  function computeSurprise(normTopk, nextToken) {
    if (!nextToken) return null;
    const hit = normTopk.find((entry) => entry.token === nextToken);
    if (!hit) return null;
    return -Math.log2(hit.prob);
  }

  function entropyColor(h) {
    const clamped = Math.max(0, Math.min(8, h || 0));
    const ratio = clamped / 8;
    const hue = (1 - ratio) * 240; // blue to red
    return `hsl(${hue}, 80%, 75%)`;
  }

  function buildTooltip(entropy, surprise, normTopk) {
    const lines = [];
    lines.push(`H ≈ ${entropy.toFixed(2)} bits`);
    if (surprise == null) {
      lines.push('S ≈ N/A');
    } else {
      lines.push(`S ≈ ${surprise.toFixed(2)} bits`);
    }
    lines.push('Top-k:');
    normTopk.forEach((entry) => {
      lines.push(`${entry.token}: ${(entry.prob * 100).toFixed(1)}%`);
    });
    return lines.join('\n');
  }

  function setupMutationObserver() {
    mutationObserver.observe(document.body, { childList: true, subtree: true });
  }

  function start() {
    initSettings().then(() => {
      scanTweets();
      setupMutationObserver();
    });
  }

  start();
})();
