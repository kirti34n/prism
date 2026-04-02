#!/usr/bin/env python3
"""
Neti — Think different. Not different answers. Different angles.

One question through multiple lenses. Measures how your thinking shifts.
Research-backed structural constraints. Zero dependencies.

  neti "question"              # explore: position → perspectives → revised position
  neti check "AI conclusion"   # challenge an AI conclusion before committing
  neti quick "question"        # just show perspectives, no measurement
  neti think                   # random prompt → explore
  neti insights                # your thinking patterns over time
  neti history                 # recent sessions
  neti config [key] [val]      # show or set configuration
  neti setup install           # make 'neti' a global command
  neti setup claude            # Claude Code (/neti slash command)
  neti setup gemini            # Gemini CLI
  neti setup all               # all tool integrations
  neti json "question"         # machine-readable output
  neti json --check "concl"    # machine-readable check output
  neti reset                   # fresh start
  neti --version               # show version

Config:  .neti.json (project) → ~/.config/neti/config.json (global) → auto-detect
"""

from __future__ import annotations

import json, sys, time, hashlib, random, re, math, os, threading
from pathlib import Path
from datetime import datetime
from collections import Counter

VERSION = 2
__version__ = '2.3.0'

# ============================================================
# CONSTANTS
# ============================================================

# Measurement thresholds
REFRAMING_DISTANCE = 0.3
RECONCEPTUALIZATION_SHIFT = 0.15
DIRECTION_THRESHOLD = 0.05
INDEPENDENCE_SCALING = 0.6
DESTABILIZATION_CONFIDENCE_DROP = 3
SHIFT_THRESHOLD = 0.1
INPUT_TRUNCATION_LIMIT = 500

# Operational limits
MAX_CONCURRENT_LLM = 8
MAX_SESSIONS = 500
OLLAMA_TIMEOUT = 120
API_TIMEOUT = 60
GEMINI_TIMEOUT = 90

# Strategy weight bounds
WEIGHT_MIN = 0.1
WEIGHT_MAX = 5.0

# ============================================================
# PATHS
# ============================================================

CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'neti'
STATE_FILE = CONFIG_DIR / 'state.json'
LOG_FILE = CONFIG_DIR / 'neti.log'


# ============================================================
# CONFIG
# ============================================================

def _find_project_config():
    d = Path.cwd()
    while True:
        f = d / '.neti.json'
        if f.exists():
            try:
                return json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        parent = d.parent
        if parent == d:
            break
        d = parent
    return {}


def _load_global_config():
    f = CONFIG_DIR / 'config.json'
    if f.exists():
        try:
            return json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_global_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / 'config.json').write_text(json.dumps(cfg, indent=2))


def _detect_provider():
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request('http://localhost:11434/api/tags')
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        models = [m['name'] for m in data.get('models', [])]
        if models:
            return {'provider': 'ollama', 'model': models[0]}
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        pass
    if os.environ.get('OPENAI_API_KEY'):
        return {'provider': 'openai', 'model': 'gpt-4o-mini'}
    if os.environ.get('ANTHROPIC_API_KEY'):
        return {'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}
    if os.environ.get('GOOGLE_API_KEY'):
        return {'provider': 'gemini', 'model': 'gemini-2.0-flash'}
    if os.environ.get('OPENROUTER_API_KEY'):
        return {'provider': 'openrouter', 'model': 'anthropic/claude-sonnet-4-20250514'}
    return None


def _load_config():
    defaults = {
        'provider': 'ollama', 'model': 'qwen3:8b', 'temperature': 0.9,
        'max_tokens': 500, 'strategies': 'auto',
        'num_perspectives': 4, 'num_shown': 3,
    }
    global_cfg = _load_global_config()
    project_cfg = _find_project_config()
    # Only auto-detect if no provider explicitly configured (avoids 3s Ollama timeout)
    if 'provider' not in global_cfg and 'provider' not in project_cfg:
        detected = _detect_provider()
        if detected:
            defaults['provider'] = detected['provider']
            defaults['model'] = detected['model']
    return {**defaults, **global_cfg, **project_cfg}


# ============================================================
# MEASUREMENT
# ============================================================

_EMBEDDER = None
_EMBEDDER_CHECKED = False
_EMB_CACHE = {}


def _load_embedder():
    global _EMBEDDER, _EMBEDDER_CHECKED
    if _EMBEDDER_CHECKED:
        return _EMBEDDER
    _EMBEDDER_CHECKED = True
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    except ImportError:
        _EMBEDDER = None
    return _EMBEDDER


def _embed(text):
    model = _load_embedder()
    if model is None:
        return None
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    if key not in _EMB_CACHE:
        _EMB_CACHE[key] = model.encode(text, normalize_embeddings=True).tolist()
    return _EMB_CACHE[key]


def _cosine_distance_emb(a: list[float], b: list[float]) -> float:
    return max(0.0, 1.0 - sum(x * y for x, y in zip(a, b)))


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _bow_distance(a: str, b: str) -> float:
    va, vb = Counter(_tokenize(a)), Counter(_tokenize(b))
    inter = set(va) & set(vb)
    if not inter:
        return 1.0
    dot = sum(va[w] * vb[w] for w in inter)
    ma = math.sqrt(sum(v ** 2 for v in va.values()))
    mb = math.sqrt(sum(v ** 2 for v in vb.values()))
    if not ma or not mb:
        return 1.0
    d = 1.0 - dot / (ma * mb)
    return max(0.0, round(d, 10))


def _distance(a: str, b: str) -> float:
    ea, eb = _embed(a), _embed(b)
    if ea is not None and eb is not None:
        return _cosine_distance_emb(ea, eb)
    return _bow_distance(a, b)


def _measurement_method():
    return 'semantic' if _load_embedder() else 'lexical'


def _measure_direction(before: str | None, after: str | None, responses: dict[str, str]) -> str:
    if not before or not after:
        return 'unknown'
    bd = {k: _distance(before, v) for k, v in responses.items() if v}
    ad = {k: _distance(after, v) for k, v in responses.items() if v}
    if not bd or not ad:
        return 'unknown'
    bn = min(bd.values())
    ank = min(ad, key=ad.get)
    an = ad[ank]
    if an < bn - DIRECTION_THRESHOLD:
        return f'toward_{ank}'
    elif an > bn + DIRECTION_THRESHOLD:
        return 'independent'
    return 'stable'


def _measure_independence(after: str | None, responses: dict[str, str]) -> float | None:
    if not after:
        return None
    dists = [_distance(after, v) for v in responses.values() if v]
    return min(1.0, min(dists) / INDEPENDENCE_SCALING) if dists else None


def _classify_session(conf_before: int | None, conf_after: int | None, shift: float | None,
                      direction: str, after_text: str | None, question: str) -> str:
    """Classify session based on confidence + text + direction signals."""
    # Reframing: user asked a different question
    is_reframing = (after_text and '?' in after_text
                    and _distance(question, after_text) > REFRAMING_DISTANCE)
    if is_reframing:
        return 'reframing'

    # Destabilization: confidence dropped significantly
    if conf_before is not None and conf_after is not None:
        if conf_after - conf_before <= -DESTABILIZATION_CONFIDENCE_DROP:
            return 'destabilization'

    # Adoption: moved toward a model response
    if direction and direction.startswith('toward_'):
        return 'adoption'

    # Reconceptualization: genuine shift in independent direction
    if shift is not None and shift > RECONCEPTUALIZATION_SHIFT and direction == 'independent':
        return 'reconceptualization'

    # Some shift but not clearly classified
    if shift is not None and shift > SHIFT_THRESHOLD:
        return 'shift'

    return 'unshaken'


# ============================================================
# LLM
# ============================================================

def _read_with_timeout(resp, timeout=60):
    result = [None]
    def _read():
        try:
            result[0] = resp.read()
        except OSError:
            result[0] = b''
    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    return result[0] if result[0] else b''


def _verbose(msg):
    if os.environ.get('NETI_VERBOSE'):
        print(f"  [verbose] {msg}", flush=True)


def _build_ollama(system_prompt, user_prompt, model, temp, max_tokens, config):
    url = config.get('endpoint', 'http://localhost:11434') + '/api/chat'
    predict = max(max_tokens * 3, 1500) if 'qwen' in model.lower() else max_tokens
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': user_prompt}],
        'stream': False,
        'options': {'temperature': temp, 'num_predict': predict}
    }).encode()
    headers = {'Content-Type': 'application/json'}
    return url, body, headers, OLLAMA_TIMEOUT


def _parse_ollama(raw):
    text = json.loads(raw).get('message', {}).get('content', '')
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _build_openai_compat(system_prompt, user_prompt, model, temp, max_tokens, config):
    provider = config.get('provider', 'openai')
    if provider == 'openai':
        url = 'https://api.openai.com/v1/chat/completions'
        key = os.environ.get('OPENAI_API_KEY', '')
    elif provider == 'openrouter':
        url = 'https://openrouter.ai/api/v1/chat/completions'
        key = os.environ.get('OPENROUTER_API_KEY', '')
    else:
        url = config.get('endpoint', 'http://localhost:1234/v1') + '/chat/completions'
        key = os.environ.get('CUSTOM_API_KEY', '')
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': user_prompt}],
        'temperature': temp, 'max_tokens': max_tokens,
    }).encode()
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    return url, body, headers, API_TIMEOUT


def _parse_openai_compat(raw):
    choices = json.loads(raw).get('choices', [])
    return choices[0].get('message', {}).get('content', '') if choices else ''


def _build_anthropic(system_prompt, user_prompt, model, temp, max_tokens, config):
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    body = json.dumps({
        'model': model, 'max_tokens': max_tokens,
        'system': system_prompt,
        'messages': [{'role': 'user', 'content': user_prompt}],
        'temperature': temp,
    }).encode()
    headers = {
        'Content-Type': 'application/json', 'x-api-key': key,
        'anthropic-version': '2023-06-01',
    }
    return 'https://api.anthropic.com/v1/messages', body, headers, API_TIMEOUT


def _parse_anthropic(raw):
    blocks = json.loads(raw).get('content', [])
    return ' '.join(b.get('text', '') for b in blocks if b.get('type') == 'text')


def _build_gemini(system_prompt, user_prompt, model, temp, max_tokens, config):
    key = os.environ.get('GOOGLE_API_KEY', '')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        'system_instruction': {'parts': [{'text': system_prompt}]},
        'contents': [{'parts': [{'text': user_prompt}]}],
        'generationConfig': {'temperature': temp, 'maxOutputTokens': max_tokens}
    }).encode()
    headers = {'Content-Type': 'application/json'}
    return url, body, headers, GEMINI_TIMEOUT


def _parse_gemini(raw):
    parts = json.loads(raw).get('candidates', [{}])[0].get('content', {}).get('parts', [])
    return parts[0].get('text', '') if parts else ''


_PROVIDERS = {
    'ollama': (_build_ollama, _parse_ollama),
    'openai': (_build_openai_compat, _parse_openai_compat),
    'openrouter': (_build_openai_compat, _parse_openai_compat),
    'custom': (_build_openai_compat, _parse_openai_compat),
    'anthropic': (_build_anthropic, _parse_anthropic),
    'gemini': (_build_gemini, _parse_gemini),
}


def _llm_call(system_prompt: str, user_prompt: str, config: dict) -> str:
    import urllib.request, urllib.error
    provider = config.get('provider', 'ollama')
    model = config.get('model', '')
    temp = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens', 300)

    build_fn, parse_fn = _PROVIDERS.get(provider, (_build_ollama, _parse_ollama))

    for attempt in range(2):
        try:
            url, body, headers, timeout = build_fn(
                system_prompt, user_prompt, model, temp, max_tokens, config)
            req = urllib.request.Request(url, data=body, headers=headers)
            resp = urllib.request.urlopen(req, timeout=timeout)
            raw = _read_with_timeout(resp, timeout=timeout)
            return parse_fn(raw)
        except Exception as e:
            err = f"{provider}/{model}: {type(e).__name__}: {e}"
            _log(f"LLM error: {err}")
            _verbose(f"LLM error (attempt {attempt+1}): {err}")
            if attempt == 0:
                time.sleep(1)
            else:
                return ''
    return ''


# ============================================================
# STRATEGIES — research-backed structural constraints
# ============================================================

STRATEGIES = {
    'default': {
        'name': 'Default',
        'system': 'Answer the question directly. Give the most practical, well-reasoned answer you can. Include specific technologies, approaches, or steps — not vague principles. If there are trade-offs, name them concretely.',
    },
    # === Evidence-backed general strategies ===
    'devils_advocate': {
        'name': "Devil's Advocate",
        'system': 'Argue AGAINST the most likely answer. No hedging, no "both sides." Take the opposing position and defend it like you believe it. Name specific real-world examples where the popular approach failed. Attack the weakest assumptions. Make the reader uncomfortable with the conventional wisdom.',
        'prefix': 'The common answer is probably obvious. Argue against it:\n\n',
        'evidence': 'Lord, Lepper & Preston 1984 — "consider the opposite" eliminates anchoring',
    },
    'blind_spot': {
        'name': 'Blind Spot',
        'system': 'Identify exactly ONE hidden assumption or overlooked constraint that changes the entire framing. Not a minor detail — a structural blind spot that, once seen, makes the obvious answer look naive. Explain the specific mechanism that keeps people from seeing it. Give a concrete example of what goes wrong when it is missed.',
        'prefix': 'What is the one thing most people miss:\n\n',
    },
    'first_principles': {
        'name': 'First Principles',
        'system': 'Strip away every inherited assumption. Name the 2-3 things "everyone knows" that might be wrong. For each, describe the specific scenario where it breaks down, cite a real precedent if possible, and show what answer you get when you rebuild without that assumption. The rebuilt answer should surprise.',
        'prefix': 'Break this down to first principles:\n\n',
        'evidence': 'Koriat 1980 — counterargument generation calibrates confidence',
    },
    'inversion': {
        'name': 'Inversion',
        'system': 'Answer the exact OPPOSITE question. If they ask how to succeed, write a detailed recipe for guaranteed failure — specific steps, specific mistakes, specific bad decisions. Be thorough and concrete. Then explicitly state what the original answer was hiding that the inversion reveals.',
        'prefix': 'Answer the opposite:\n\n',
        'evidence': 'Mussweiler 2000 — considering the opposite eliminates anchoring in expert judgment',
    },
    'systems': {
        'name': 'Systems',
        'system': 'Ignore the direct answer. Map the second-order and third-order effects. What breaks, shifts, or emerges BECAUSE of the obvious answer? Follow each causal chain at least three steps with specific examples. Name the feedback loops. Identify where the system will fight back against the intended change.',
        'prefix': 'What are the downstream consequences:\n\n',
    },
    'stakeholder': {
        'name': 'Stakeholder',
        'system': 'Identify who gets harmed, marginalized, or locked out by the conventional approach. Tell the story entirely from their perspective — their constraints, their frustrations, their alternatives. Name specific scenarios. Make the reader feel the friction the standard answer creates for someone who is not the assumed user.',
        'prefix': 'Who loses when we answer this the standard way:\n\n',
        'evidence': 'Galinsky & Moskowitz 2000 — perspective-taking reduces bias',
    },
    # === Research-specific strategies (evidence-backed) ===
    'pre_mortem': {
        'name': 'Pre-Mortem',
        'system': 'It is 18 months from now. This approach was pursued and FAILED. The failure was predictable in hindsight. Write the post-mortem: name the specific failure mode, the early warning signs that were rationalized away, the moment the team should have pivoted but did not. Be brutally concrete — "the API rate limits hit at 10K users and there was no fallback" not "scalability was an issue."',
        'prefix': 'Assume this has already failed:\n\n',
        'evidence': 'Klein 2007 — prospective hindsight generates 30% more failure reasons, reduces overconfidence',
    },
    'alternative_hypothesis': {
        'name': 'Alt Hypothesis',
        'system': 'Name 3 genuinely different explanations or approaches for the same problem. Not variations on a theme — structurally different mechanisms or architectures. For each, state (a) the core insight that makes it work, (b) one specific scenario where it outperforms the obvious answer, and (c) the exact test or metric that would distinguish between them.',
        'prefix': 'What else could explain this:\n\n',
        'evidence': 'Hirt & Markman 1995 — any alternative triggers debiasing simulation mindset',
    },
    'falsification': {
        'name': 'Falsification',
        'system': 'Design the exact test that would DISPROVE this approach. Name the specific metric, threshold, and scenario. "If X does not achieve Y under condition Z within timeframe W, this approach is wrong." If no test can disprove it, that itself is a red flag — explain exactly why unfalsifiable plans are dangerous and what would make this one testable.',
        'prefix': 'What would disprove this:\n\n',
        'evidence': 'Tetlock 2015 — superforecasters are 60% more accurate; falsification thinking is their key habit',
    },
    'adjacent_field': {
        'name': 'Adjacent Field',
        'system': 'Pick a specific field that has already solved an analogous problem. Name the field, the specific technique or framework, and how it maps onto this problem in concrete terms. Use their vocabulary. Describe what a practitioner from that field would do differently in the first week of this project. The reframing should produce at least one specific, actionable idea the original framing would never generate.',
        'prefix': 'How would a different field see this:\n\n',
        'evidence': 'Uzzi 2013 (Science, 17.9M papers) — atypical combinations produce 2x citation impact',
    },
}

STRATEGY_KEYS = [k for k in STRATEGIES if k != 'default']

# Fixed set for `check` command
CHECK_STRATEGIES = ['pre_mortem', 'alternative_hypothesis', 'falsification', 'blind_spot']


def _select_strategies(state, config):
    strategies_cfg = config.get('strategies', 'auto')
    n = config.get('num_perspectives', 4)
    if isinstance(strategies_cfg, list):
        valid = [k for k in strategies_cfg if k in STRATEGIES and k != 'default']
        if valid:
            return valid[:n]
    weights = state.get('strategy_weights', {})
    scored = [(k, weights.get(k, 1.0)) for k in STRATEGY_KEYS]
    # Cold start: when all weights are equal, fully randomize
    unique_weights = set(w for _, w in scored)
    if len(unique_weights) <= 1:
        return random.sample([k for k, _ in scored], min(n, len(scored)))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s[0] for s in scored[:min(2, len(scored))]]
    remaining = [s[0] for s in scored if s[0] not in top]
    n_explore = min(n - len(top), len(remaining))
    explore_picks = random.sample(remaining, n_explore) if n_explore > 0 else []
    selected = (top + explore_picks)[:n]
    # Force at least 1 from bottom half (convergence protection)
    bottom_half = [s[0] for s in scored[len(scored)//2:]]
    if not any(s in bottom_half for s in selected):
        selected[-1] = random.choice(bottom_half)
    return selected


def _generate_perspectives(question, strategies, config, quiet=False):
    provider = config.get('provider', 'ollama')
    results = {}
    _p = (lambda *a, **kw: None) if quiet else (lambda *a, **kw: print(*a, **kw))

    default_resp = _llm_call(STRATEGIES['default']['system'], question,
        {**config, 'max_tokens': config.get('max_tokens', 300) + 100})
    results['default'] = default_resp
    _p(f"\r  Generating... [1/{len(strategies)+1}]", end="", flush=True)

    lock = threading.Lock()
    sem = threading.Semaphore(MAX_CONCURRENT_LLM)
    completed = [0]
    total = len(strategies) + 1
    def _gen(idx, key):
        with sem:
            s = STRATEGIES[key]
            contrastive = ''
            if default_resp:
                contrastive = f"The conventional AI answer is:\n{default_resp[:400]}\n\nNow challenge that specific answer.\n\n"
            resp = _llm_call(s['system'], contrastive + s.get('prefix', '') + question, config)
            with lock:
                results[key] = resp
                completed[0] += 1
                _p(f"\r  Generating... [{completed[0]}/{total}]", end="", flush=True)
    threads = []
    for i, key in enumerate(strategies):
        t = threading.Thread(target=_gen, args=(i, key), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=120)
    if not quiet:
        print()

    return results


# ============================================================
# STATE
# ============================================================

def _new_state():
    return {
        'version': VERSION,
        'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:12],
        'created': datetime.now().isoformat(),
        'strategy_weights': {k: 1.0 for k in STRATEGY_KEYS},
        'sessions': [],
    }


def _migrate_legacy_state():
    # Migration from ~/.config/prism/ (old config dir)
    old_config_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'prism'
    old_state = old_config_dir / 'state.json'
    if old_state.exists() and not STATE_FILE.exists():
        try:
            data = json.loads(old_state.read_text())
            if data.get('version') == VERSION:
                _save_state(data)
                return data
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    # Migration from local prism_state.json (very old format)
    legacy = Path(__file__).parent / 'prism_state.json'
    if legacy.exists() and not STATE_FILE.exists():
        try:
            old = json.loads(legacy.read_text())
            new = _new_state()
            new['sessions'] = old.get('sessions', [])
            for k in STRATEGY_KEYS:
                w = old.get('strategy_weights', {}).get(k)
                if w:
                    new['strategy_weights'][k] = w
            _save_state(new)
            return new
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    return None


def _load_state():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if data.get('version') == VERSION:
                return data
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    migrated = _migrate_legacy_state()
    return migrated if migrated else _new_state()


def _save_state(state):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    state['last_used'] = datetime.now().isoformat()
    sessions = state.get('sessions', [])
    if len(sessions) > MAX_SESSIONS:
        state['sessions'] = sessions[-MAX_SESSIONS:]
    tmp = STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def _log(msg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# ============================================================
# FEEDBACK
# ============================================================

def _update_weights(state: dict, shown: list[str], rated: str | None,
                    session_type: str, direction: str) -> None:
    w = state.get('strategy_weights', {})
    for key in shown:
        w[key] = w.get(key, 1.0) * 1.02
    if rated and rated in w:
        w[rated] = w[rated] * 1.2
        for key in shown:
            if key != rated:
                w[key] = w.get(key, 1.0) * 0.95
    if session_type in ('reframing', 'reconceptualization', 'destabilization'):
        for key in shown:
            w[key] = w.get(key, 1.0) * 1.1
    elif session_type == 'adoption':
        for key in shown:
            w[key] = w.get(key, 1.0) * 0.98
    elif session_type == 'unshaken':
        for key in shown:
            w[key] = w.get(key, 1.0) * 0.95
    if direction and direction.startswith('toward_'):
        target = direction.replace('toward_', '')
        if target in w:
            w[target] = w.get(target, 1.0) * 1.05
    # Cap weight ratio: no strategy exceeds 3x the lowest
    if w:
        min_w = min(w.values())
        for key in w:
            w[key] = min(w[key], min_w * 3.0)
    # Every 25 sessions, pull extreme weights 20% toward mean
    n_sessions = len(state.get('sessions', []))
    if n_sessions > 0 and n_sessions % 25 == 0 and w:
        mean_w = sum(w.values()) / len(w)
        for key in w:
            w[key] = w[key] * 0.8 + mean_w * 0.2
    # Normalize: weights average to 1.0, clamp to [0.1, 5.0]
    total = sum(w.values())
    n_keys = len(w)
    if total > 0 and n_keys > 0:
        for key in w:
            w[key] = max(WEIGHT_MIN, min(WEIGHT_MAX, w[key] * n_keys / total))
    state['strategy_weights'] = w


# ============================================================
# COMMANDS
# ============================================================

PROMPTS = [
    "What are you avoiding right now?",
    "What do you believe that you can't prove?",
    "What assumption are you making that might be wrong?",
    "What would you do if you knew it would fail?",
    "What have you been wrong about recently?",
    "What decision are you putting off?",
    "What would disprove your current hypothesis?",
    "What would change if you stopped?",
    "What are you pretending isn't a problem?",
    "What are you optimizing for, and should you be?",
    "Who would disagree with your approach, and why?",
    "What question should you be asking instead?",
    "What's the thing you keep coming back to?",
    "What are you building and why?",
    "What would a researcher from a different field say about your work?",
]


def _read_confidence():
    if not sys.stdin.isatty():
        return None
    try:
        val = input("  Confidence (1-10): ").strip()
        if val and val.isdigit():
            n = int(val)
            if 1 <= n <= 10:
                return n
    except (EOFError, KeyboardInterrupt):
        pass
    return None


def explore(question):
    """Position → perspectives → revised position → measure."""
    state = _load_state()
    cfg = _load_config()

    print(f"\n  NETI")
    print(f"  {'=' * 56}")
    print(f"  {question}")
    print(f"  {'=' * 56}\n")

    # Before: position + confidence
    print("  Your position (before seeing anything):", flush=True)
    human_before = None
    if sys.stdin.isatty():
        try:
            human_before = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            pass
    if not human_before:
        human_before = None
    conf_before = _read_confidence() if human_before else None

    # Generate
    print(f"\n  Generating perspectives ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    strategies = _select_strategies(state, cfg)
    w = state.get('strategy_weights', {})
    _verbose(f"strategies: {[f'{s}({w.get(s, 1.0):.2f})' for s in strategies]}")
    responses = _generate_perspectives(question, strategies, cfg)
    responses = {k: v for k, v in responses.items() if v}

    if 'default' not in responses:
        print(f"  Default response failed ({cfg.get('provider')}/{cfg.get('model')}).")
        print(f"  Run 'neti config' to verify settings.")
        return

    non_default = {k: v for k, v in responses.items() if k != 'default'}
    if not non_default:
        print(f"\n  {'─' * 56}\n  DEFAULT\n  {'─' * 56}")
        _print_wrapped(responses['default'], indent=4)
        return

    # Rank by divergence
    divergences = {k: _distance(responses['default'], v) for k, v in non_default.items()}
    shown_keys = sorted(divergences, key=divergences.get, reverse=True)[:cfg.get('num_shown', 3)]

    # Show
    print(f"\n  {'─' * 56}")
    print("  DEFAULT ANSWER")
    print(f"  {'─' * 56}")
    _print_wrapped(responses['default'], indent=4)

    for i, key in enumerate(shown_keys):
        s = STRATEGIES.get(key, {})
        print(f"\n  {'─' * 56}")
        print(f"  {i+1}. {s.get('name', key).upper()} (distance: {divergences[key]:.3f})")
        print(f"  {'─' * 56}")
        _print_wrapped(responses[key], indent=4)

    # After: revised position + confidence
    print(f"\n  {'─' * 56}")
    print("  Same position? Changed? Or a different question entirely:", flush=True)
    human_after = None
    if sys.stdin.isatty():
        try:
            human_after = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            pass
    if not human_after:
        human_after = None
    conf_after = _read_confidence() if human_after else None

    # Measure
    shift = _distance(human_before, human_after) if (human_before and human_after) else None
    direction = _measure_direction(human_before, human_after, responses)
    independence = _measure_independence(human_after, responses)
    convergence = _distance(human_after, responses['default']) if human_after else None
    session_type = _classify_session(conf_before, conf_after, shift, direction,
                                      human_after, question)

    if (human_before and len(human_before) > INPUT_TRUNCATION_LIMIT) or (human_after and len(human_after) > INPUT_TRUNCATION_LIMIT):
        print("  (Note: response truncated to 500 characters for storage)")

    session = {
        'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:8],
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'human_before': human_before[:INPUT_TRUNCATION_LIMIT] if human_before else None,
        'human_after': human_after[:INPUT_TRUNCATION_LIMIT] if human_after else None,
        'confidence_before': conf_before,
        'confidence_after': conf_after,
        'strategies_shown': shown_keys,
        'divergences': {k: round(v, 4) for k, v in divergences.items()},
        'shift': round(shift, 4) if shift is not None else None,
        'direction': direction,
        'independence': round(independence, 4) if independence is not None else None,
        'convergence_score': round(convergence, 4) if convergence is not None else None,
        'session_type': session_type,
        'user_rating': None,
        'measurement': _measurement_method(),
    }

    # Show measurement
    if human_before and human_after:
        print(f"\n  {'=' * 56}")
        print(f"  MEASUREMENT ({_measurement_method()})")
        print(f"  {'=' * 56}")

        # Session type
        type_desc = {
            'reframing': 'You asked a different question — frame shift',
            'reconceptualization': 'Genuine new thinking — independent direction',
            'destabilization': 'Productive doubt — confidence shaken',
            'adoption': 'Moved toward a model response — check if genuine',
            'shift': 'Some change detected',
            'unshaken': 'No significant change',
        }
        print(f"\n  Session: {session_type}")
        print(f"  {type_desc.get(session_type, '')}")

        if conf_before is not None and conf_after is not None:
            delta = conf_after - conf_before
            print(f"  Confidence: {conf_before} → {conf_after} ({delta:+d})")
        if shift is not None:
            print(f"  Text shift: {shift:.4f}")
        if direction != 'unknown':
            if direction.startswith('toward_'):
                sname = STRATEGIES.get(direction.replace('toward_', ''), {}).get('name', direction)
                print(f"  Direction: toward {sname}")
            else:
                print(f"  Direction: {direction}")
        if independence is not None:
            print(f"  Independence: {independence:.0%}")

    # Feedback
    rated = None
    if sys.stdin.isatty() and shown_keys:
        try:
            parts = [f"{i+1}={STRATEGIES.get(k, {}).get('name', k)}" for i, k in enumerate(shown_keys)]
            print(f"\n  Most useful? ({', '.join(parts)}, Enter to skip)")
            choice = input("  > ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(shown_keys):
                rated = shown_keys[int(choice) - 1]
                session['user_rating'] = rated
        except (EOFError, KeyboardInterrupt):
            pass

    _update_weights(state, shown_keys, rated, session_type, direction)
    state.setdefault('sessions', []).append(session)
    _save_state(state)

    n = len(state['sessions'])
    if n < 5:
        print(f"\n  Session logged ({n}/5 for first insights).\n")
    else:
        print(f"\n  Session logged. Run 'neti insights' for patterns.\n")
    _log(f"explore: q='{question[:50]}' type={session_type} conf={conf_before}→{conf_after}")


def check(conclusion):
    """Challenge an AI conclusion before committing."""
    cfg = _load_config()

    print(f"\n  NETI — Challenge")
    print(f"  {'=' * 56}")
    _print_wrapped(conclusion, indent=2)
    print(f"  {'=' * 56}\n")

    print(f"  Generating challenges ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    results = {}
    lock = threading.Lock()
    completed = [0]
    total = len(CHECK_STRATEGIES)
    def _gen(key):
        s = STRATEGIES[key]
        resp = _llm_call(s['system'], s.get('prefix', '') + conclusion, cfg)
        with lock:
            results[key] = resp
            completed[0] += 1
            print(f"\r  Generating... [{completed[0]}/{total}]", end="", flush=True)
    threads = [threading.Thread(target=_gen, args=(k,), daemon=True)
               for k in CHECK_STRATEGIES]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    print()

    for key in CHECK_STRATEGIES:
        if key in results and results[key]:
            s = STRATEGIES.get(key, {})
            print(f"\n  {'─' * 56}")
            print(f"  {s.get('name', key).upper()}")
            print(f"  {'─' * 56}")
            _print_wrapped(results[key], indent=4)

    print(f"\n  Does the conclusion still hold?\n")
    _log(f"check: '{conclusion[:50]}'")


def quick(question):
    state = _load_state()
    cfg = _load_config()
    print(f"\n  NETI — Quick View\n  {question}\n")
    strategies = _select_strategies(state, cfg)
    print(f"  Generating ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    responses = _generate_perspectives(question, strategies, cfg)
    responses = {k: v for k, v in responses.items() if v}
    if 'default' not in responses:
        print(f"  Failed ({cfg.get('provider')}/{cfg.get('model')}). Run 'neti config' to verify.")
        return
    divergences = {k: _distance(responses['default'], v)
                   for k, v in responses.items() if k != 'default'}
    sorted_keys = sorted(divergences, key=divergences.get, reverse=True)
    print(f"\n  {'─' * 56}\n  DEFAULT\n  {'─' * 56}")
    _print_wrapped(responses['default'], indent=4)
    for key in sorted_keys[:cfg.get('num_shown', 3)]:
        s = STRATEGIES.get(key, {})
        div = f" ({divergences[key]:.3f})" if key in divergences else ""
        print(f"\n  {'─' * 56}\n  {s.get('name', key).upper()}{div}\n  {'─' * 56}")
        _print_wrapped(responses[key], indent=4)
    print(f"\n  What do you think?\n")
    _log(f"quick: q='{question[:50]}'")


def think():
    print("\n  What's on your mind? (or Enter for a prompt)", flush=True)
    question = None
    if sys.stdin.isatty():
        try:
            question = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            pass
    if not question:
        question = random.choice(PROMPTS)
        print(f"  -> {question}")
    explore(question)


def insights():
    state = _load_state()
    sessions = state.get('sessions', [])
    print(f"\n  NETI — Insights\n  {'─' * 40}\n  Sessions: {len(sessions)}")

    if len(sessions) < 3:
        print(f'\n  Need at least 3 sessions.\n  Run: neti "your question"\n')
        return

    # Session type distribution
    types = Counter(s.get('session_type') for s in sessions if s.get('session_type'))
    if types:
        desc = {'reframing': 'frame shift', 'reconceptualization': 'new thinking',
                'destabilization': 'doubt', 'adoption': 'toward AI',
                'shift': 'some change', 'unshaken': 'no change'}
        print(f"\n  Session types:")
        for t, count in types.most_common():
            print(f"    {t:>20}: {count}  ({desc.get(t, t)})")

    # Confidence trends
    pairs = [(s['confidence_before'], s['confidence_after']) for s in sessions
             if s.get('confidence_before') is not None and s.get('confidence_after') is not None]
    if pairs:
        deltas = [a - b for b, a in pairs]
        avg = sum(deltas) / len(deltas)
        print(f"\n  Confidence change: {avg:+.1f} average ({len(pairs)} measured)")
        if avg < -1:
            print("    Neti is creating productive doubt")
        elif avg > 1:
            print("    Caution: confidence rising after perspectives")

    # Strategy effectiveness by deep shift rate
    if len(sessions) >= 3:
        strat_types = {}
        for s in sessions:
            st = s.get('session_type', '')
            for key in s.get('strategies_shown', []):
                strat_types.setdefault(key, []).append(st)
        if strat_types:
            print(f"\n  What challenges you (deep shift rate):")
            ranked = []
            for key, tl in strat_types.items():
                deep = sum(1 for t in tl if t in ('reframing', 'reconceptualization', 'destabilization'))
                ranked.append((key, deep / len(tl) if tl else 0, len(tl)))
            ranked.sort(key=lambda x: x[1], reverse=True)
            for key, rate, total in ranked[:6]:
                name = STRATEGIES.get(key, {}).get('name', key)
                bar = '#' * int(rate * 20)
                print(f"    {name:>20}: {rate:.0%} ({total} sessions) |{bar}|")

    # Independence
    indeps = [s['independence'] for s in sessions if s.get('independence') is not None]
    if indeps:
        print(f"\n  Independence: {sum(indeps)/len(indeps):.0%} average")

    # Convergence
    conv = [s['convergence_score'] for s in sessions if s.get('convergence_score') is not None]
    if len(conv) >= 5:
        recent = conv[-20:]
        nr = len(recent)
        xm = (nr - 1) / 2
        ym = sum(recent) / nr
        num = sum((i - xm) * (s - ym) for i, s in enumerate(recent))
        den = sum((i - xm) ** 2 for i in range(nr))
        slope = num / den if den > 0 else 0
        caveat = "  (low sample size)" if nr < 10 else ""
        print(f"\n  Convergence (last {nr} sessions):{caveat}")
        if slope < -0.01:
            print(f"    CONVERGING (slope: {slope:.4f}) — moving toward AI defaults")
        elif slope > 0.01:
            print(f"    DIVERGING (slope: {slope:.4f}) — increasingly independent")
        else:
            print(f"    STABLE (slope: {slope:.4f})")

    print()


def history(count=10):
    state = _load_state()
    sessions = state.get('sessions', [])
    print(f"\n  NETI — History ({len(sessions)} sessions)\n  {'─' * 56}")
    if not sessions:
        print('  No sessions yet.\n')
        return
    for s in sessions[-count:]:
        cb, ca = s.get('confidence_before'), s.get('confidence_after')
        if cb is not None and ca is not None:
            conf = f"{cb}→{ca}"
        else:
            sh = s.get('shift')
            conf = f"{sh:.2f}" if sh is not None else " -  "
        stype = (s.get('session_type') or s.get('shift_label') or '')[:14]
        print(f"  [{conf:>5}] {stype:>14} | {s['question'][:35]}")
    print()


def config_cmd(args):
    global_cfg = _load_global_config()
    project_cfg = _find_project_config()
    merged = _load_config()
    if not args:
        print(f"\n  NETI — Configuration\n  {'─' * 40}")
        for k, v in merged.items():
            src = '(project)' if k in project_cfg else '(global)' if k in global_cfg else '(auto)'
            print(f"  {k:>20}: {v}  {src}")
        print(f"\n  Set: neti config <key> <value>")
        print(f"  Global: {CONFIG_DIR / 'config.json'}")
        print(f"  Project: .neti.json\n")
        w = _load_state().get('strategy_weights', {})
        mode = merged.get('strategies', 'auto')
        print(f"  Strategies ({mode} mode):")
        for k in STRATEGY_KEYS:
            ev = STRATEGIES[k].get('evidence', '')
            ev_str = f"  [{ev[:40]}]" if ev else ""
            print(f"    {k:>22} ({STRATEGIES[k]['name']}) w={w.get(k, 1.0):.2f}{ev_str}")
        print(f"\n  Measurement: {_measurement_method()}")
        if _measurement_method() == 'lexical':
            print("  pip install sentence-transformers for semantic\n")
        return
    key = args[0]
    if len(args) < 2:
        print(f"  {key}: {merged.get(key, '(not set)')}")
        return

    val = ' '.join(args[1:])
    if key == 'temperature':
        val = float(val)
    elif key in ('max_tokens', 'num_perspectives', 'num_shown'):
        val = int(val)
    elif key == 'strategies':
        if val != 'auto':
            val = [s.strip() for s in val.split(',')]

    # Validate
    _VALID_PROVIDERS = ('ollama', 'openai', 'anthropic', 'gemini', 'openrouter', 'custom')
    _VALIDATORS = {
        'temperature': lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 2.0,
        'max_tokens': lambda v: isinstance(v, int) and 50 <= v <= 4096,
        'num_perspectives': lambda v: isinstance(v, int) and 1 <= v <= 10,
        'num_shown': lambda v: isinstance(v, int) and 1 <= v <= 10,
        'provider': lambda v: v in _VALID_PROVIDERS,
    }
    _RANGES = {
        'temperature': '0.0-2.0',
        'max_tokens': '50-4096',
        'num_perspectives': '1-10',
        'num_shown': '1-10',
        'provider': ', '.join(_VALID_PROVIDERS),
    }
    if key in _VALIDATORS and not _VALIDATORS[key](val):
        print(f"  Invalid value for {key}. Valid: {_RANGES[key]}")
        return

    # Ask user: global or project-level?
    target = 'global'
    if sys.stdin.isatty():
        try:
            choice = input("  Save to (g)lobal or (p)roject? [g]: ").strip().lower()
            if choice in ('p', 'project'):
                target = 'project'
        except (EOFError, KeyboardInterrupt):
            pass

    if target == 'project':
        proj_file = Path.cwd() / '.neti.json'
        proj = {}
        if proj_file.exists():
            try:
                proj = json.loads(proj_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        proj[key] = val
        proj_file.write_text(json.dumps(proj, indent=2))
        print(f"  {key} = {val}  (project: {proj_file})")
    else:
        global_cfg[key] = val
        _save_global_config(global_cfg)
        print(f"  {key} = {val}  (global: {CONFIG_DIR / 'config.json'})")


def reset():
    if sys.stdin.isatty():
        try:
            if input("  Delete all history? (yes/no): ").strip().lower() != 'yes':
                print("  Cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            return
    _save_state(_new_state())
    cfg = _load_config()
    print(f"\n  Reset. Provider: {cfg['provider']} | Model: {cfg['model']}\n")


# ============================================================
# UTILITIES
# ============================================================

def _print_wrapped(text, indent=4, width=76):
    prefix = ' ' * indent
    for line in text.split('\n'):
        if len(line) + indent <= width:
            print(f"{prefix}{line}")
        else:
            words = line.split()
            cur = prefix
            for w in words:
                if len(cur) + len(w) + 1 > width:
                    print(cur)
                    cur = prefix + w
                else:
                    cur = cur + ' ' + w if cur != prefix else cur + w
            if cur.strip():
                print(cur)


# ============================================================
# PROGRAMMATIC API
# ============================================================

def get_perspectives(question: str, n: int | None = None) -> dict:
    state = _load_state()
    cfg = _load_config()
    strategies = _select_strategies(state, cfg)
    if n is not None:
        strategies = strategies[:n]
    responses = _generate_perspectives(question, strategies, cfg, quiet=True)
    responses = {k: v for k, v in responses.items() if v}
    if 'default' not in responses:
        return {'error': 'LLM call failed'}
    result = {'question': question, 'default': responses['default'], 'perspectives': []}
    for key in strategies:
        if key in responses:
            result['perspectives'].append({
                'strategy': key,
                'name': STRATEGIES.get(key, {}).get('name', key),
                'text': responses[key],
                'divergence': round(_distance(responses['default'], responses[key]), 4),
            })
    result['perspectives'].sort(key=lambda x: x['divergence'], reverse=True)
    return result


def get_check(conclusion: str) -> dict:
    """Challenge a conclusion. Returns structured JSON for integrations."""
    cfg = _load_config()
    provider = cfg.get('provider', 'ollama')
    results = {}

    lock = threading.Lock()
    def _gen(key):
        s = STRATEGIES[key]
        resp = _llm_call(s['system'], s.get('prefix', '') + conclusion, cfg)
        with lock:
            if resp:
                results[key] = resp
    threads = [threading.Thread(target=_gen, args=(k,), daemon=True)
               for k in CHECK_STRATEGIES]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    output = {'conclusion': conclusion, 'challenges': []}
    for key in CHECK_STRATEGIES:
        if key in results:
            output['challenges'].append({
                'strategy': key,
                'name': STRATEGIES[key]['name'],
                'text': results[key],
            })
    return output


# ============================================================
# INTEGRATION SETUP
# ============================================================

_DETECT = {
    'claude': Path.home() / '.claude',
    'cursor': Path.home() / '.cursor',
    'codex': Path.home() / '.codex',
    'windsurf': Path.home() / '.windsurf',
    'kiro': Path.home() / '.kiro',
    'gemini': Path.home() / '.gemini',
    'augment': Path.home() / '.augment',
    'copilot': Path.home() / '.config' / 'github-copilot',
}


def setup(platform):
    """Set up integration with AI coding tools."""
    neti_path = os.path.realpath(__file__)

    platforms = {
        'install': lambda: _setup_install(neti_path),
        'claude': _setup_claude_code,
        'claude-code': _setup_claude_code,
        'codex': lambda: _setup_tool('codex'),
        'cursor': lambda: _setup_tool('cursor'),
        'copilot': lambda: _setup_tool('copilot'),
        'windsurf': lambda: _setup_tool('windsurf'),
        'kiro': lambda: _setup_tool('kiro'),
        'gemini': lambda: _setup_tool('gemini'),
        'augment': lambda: _setup_tool('augment'),
    }

    if platform in platforms:
        platforms[platform]()
    elif platform == 'all':
        for name, fn in platforms.items():
            if name not in ('install', 'claude-code'):
                fn()
        print(f"\n  All integrations installed.\n")
    elif not platform:
        # Auto-detect installed tools (interactive only)
        if not sys.stdin.isatty():
            _setup_help()
            return
        detected = [n for n, p in _DETECT.items() if p.exists()]
        if not detected:
            _setup_help()
            return
        print(f"\n  NETI — Setup")
        print(f"  {'─' * 30}")
        print(f"  Detected tools:")
        for name in detected:
            print(f"    ✓ {name.title()}")
        print()
        try:
            choice = input("  Set up all detected tools? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = 'n'
        if choice in ('y', 'yes'):
            for name in detected:
                if name in platforms:
                    platforms[name]()
            print(f"\n  Done.\n")
        else:
            print(f"  Cancelled.\n")
    else:
        _setup_help()


def _setup_help():
    print(f"\n  NETI — Setup")
    print(f"  {'─' * 40}")
    print(f"  neti setup install    # make 'neti' available globally")
    print(f"  neti setup claude     # Claude Code (/neti, /neti-check)")
    print(f"  neti setup codex      # Codex CLI")
    print(f"  neti setup cursor     # Cursor")
    print(f"  neti setup copilot    # GitHub Copilot")
    print(f"  neti setup windsurf   # Windsurf")
    print(f"  neti setup kiro       # Kiro")
    print(f"  neti setup gemini     # Gemini CLI")
    print(f"  neti setup augment    # Augment Code")
    print(f"  neti setup all        # all of the above\n")


def _setup_install(neti_path):
    """Make 'neti' available as a system command."""
    import shutil
    existing = shutil.which('neti')
    resolved = os.path.realpath(existing) if existing else ''

    # If already installed via pip/pipx, skip symlink
    if existing and ('pipx' in resolved or 'site-packages' in resolved):
        print(f"\n  Already installed via pip/pipx: {existing}")
        return

    bin_dir = Path.home() / '.local' / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / 'neti'

    # Make executable
    os.chmod(neti_path, os.stat(neti_path).st_mode | 0o755)

    # Create symlink
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(neti_path)

    # Check PATH
    on_path = str(bin_dir) in os.environ.get('PATH', '').split(':')

    print(f"\n  Installed: {link} -> {neti_path}")
    if on_path:
        print(f"  'neti' is on PATH. Ready to use.")
    else:
        print(f"\n  Add to ~/.bashrc or ~/.zshrc:")
        print(f'    export PATH="$PATH:{bin_dir}"')
        print(f"  Then: source ~/.bashrc")


# ---- Embedded Claude Code skill content (no external files needed) ----

_SKILL_NETI = """\
---
name: neti
description: Generate divergent perspectives on a question using research-backed cognitive strategies. Reveals how AI shapes your thinking.
argument-hint: <question or topic>
---

# Neti — Divergent Perspectives

Generate structurally different perspectives using 10 research-backed cognitive strategies.

## When to use

- User asks for perspectives, different angles, challenges thinking, says "neti"
- User is making a decision or evaluating approaches
- User wants to stress-test a technical approach

## How to run

**Path 1 — CLI available:** Run `neti json "$ARGUMENTS"`, parse the JSON output. It includes divergence scores and tracking. Present using the format in Step 4.

**Path 2 — No CLI:** Generate natively using the steps below.

### Step 1: Default Answer

Give the most practical, specific answer to the question. 3-4 sentences. Name specific technologies, approaches, or steps — not vague principles.

### Step 2: Generate 3 Perspectives

Pick 3 strategies from the table below. Each MUST follow its structural constraint exactly. Do NOT hedge or qualify — commit fully to the perspective.

| Strategy | Constraint |
|----------|-----------|
| Devil's Advocate | Argue AGAINST the common position. No hedging. Real failure examples. |
| Pre-Mortem | 18 months from now, this failed. Write the post-mortem. Specific failure modes. |
| Falsification | Design the exact test that would disprove this. Metric, threshold, timeframe. |
| Blind Spot | ONE hidden assumption that changes the entire framing. The mechanism that hides it. |
| Alt Hypothesis | 3 structurally different explanations. Core insight, scenario where it wins, distinguishing test. |
| First Principles | List 2-3 "everyone knows" assumptions. Show where each breaks. Rebuild without them. |
| Inversion | Answer the exact opposite question in detail. What does the inversion reveal? |
| Systems | Only 2nd/3rd order effects. Follow causal chains 3 steps. Name feedback loops. |
| Stakeholder | Who gets harmed? Tell the story from their perspective. Make the friction concrete. |
| Adjacent Field | Pick a specific field that solved an analogous problem. Map their technique onto this. |

### Step 3: Rank by Divergence

Order perspectives by how different each is from the default. Most divergent first.

### Step 4: Present

**Default Answer**
[the default — 3-4 sentences]

---

**Divergent Perspectives**

**1. [Strategy Name]**
[Full perspective text — 4-8 sentences minimum]

**2. [Strategy Name]**
[Full perspective text]

**3. [Strategy Name]**
[Full perspective text]

---

*Do any of these shift how you're thinking about this?*

## Important

Do NOT paraphrase, shorten, or editorialize perspectives. The value is in the specifics, examples, and concrete details. Each perspective should commit fully to its position — no "on the other hand" hedging. If a perspective names specific technologies, failure modes, or examples, keep them all.
"""

_SKILL_CHECK = """\
---
name: neti-check
description: Challenge a conclusion before committing to it. Generates Pre-Mortem, Alt Hypothesis, Falsification, and Blind Spot challenges.
argument-hint: <conclusion to challenge>
---

# Neti Check — Challenge a Conclusion

Stress-test a conclusion using 4 research-backed strategies before committing to it.

## When to use

- User wants to challenge or stress-test a conclusion
- User says "check this", "challenge this", "is this right", "neti check"
- User is about to commit to a technical decision based on AI advice
- User has a claim or assumption they want tested

## How to run

**Path 1 — CLI available:** Run `neti json --check "$ARGUMENTS"`, parse JSON, present results.

**Path 2 — No CLI:** Generate all 4 challenges below natively.

### Generate 4 Challenges

Apply ALL of these to the conclusion:

**Pre-Mortem** — It is 18 months from now. This approach was pursued and FAILED. The failure was predictable in hindsight. Write the post-mortem: the specific failure mode, the early warning signs that were rationalized away, the moment the team should have pivoted. Be brutally concrete.

**Alt Hypothesis** — Name 3 genuinely different explanations or approaches. Not variations on a theme — structurally different mechanisms. For each: (a) the core insight that makes it work, (b) one scenario where it outperforms the conclusion, (c) the test that distinguishes them.

**Falsification** — Design the exact test that would DISPROVE this. Name the specific metric, threshold, and scenario. If no test can disprove it, explain why that's a red flag.

**Blind Spot** — Identify exactly ONE hidden assumption that changes the entire framing. Not a minor detail — a structural blind spot that, once seen, makes the conclusion look naive. Explain the mechanism that keeps people from seeing it.

### Present

For each challenge:
1. **Strategy name** in bold
2. The full challenge (4-8 sentences minimum — commit fully, no hedging)

End with: *"Does the original conclusion still hold?"*

## Important

Do NOT soften or balance challenges. The entire point is to stress-test. Each challenge should make the user uncomfortable with the conclusion — that's the signal it's working.
"""

_SKILL_AUTO = """\
---
name: neti-auto
description: Automatically add diverse perspectives when user is making decisions, evaluating approaches, or asking should-I questions
user-invocable: false
---

# Neti Auto — Lightweight Perspective Nudges

When you detect the user is making a decision or evaluating approaches, add 2 brief divergent perspectives inline.

## When to trigger

- "should we/I...", "which approach...", "is it better to..."
- Evaluating trade-offs between options
- About to commit to an architecture or approach

Do NOT trigger on simple factual questions, implementation requests, or when the user is asking for help with something already decided.

## How to generate

Pick 2 strategies from: Devil's Advocate, Pre-Mortem, Blind Spot, First Principles, Inversion, Systems.

For each, write 2-3 sentences that commit fully to that perspective. No hedging.

## How to present

After your normal response, add:

---

**Neti perspectives:**
- **[Strategy]:** [2-3 sentence perspective]
- **[Strategy]:** [2-3 sentence perspective]

*Run /neti for the full experience with more perspectives.*

---

Keep it lightweight. This is a nudge, not the full Neti flow.
"""


def _setup_claude_code():
    """Generate and install Neti as a Claude Code plugin (works from any install method)."""
    claude_dir = Path.home() / '.claude'
    claude_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Register in settings.json ---
    settings_path = claude_dir / 'settings.json'
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if 'extraKnownMarketplaces' not in settings:
        settings['extraKnownMarketplaces'] = {}
    settings['extraKnownMarketplaces']['neti-skill'] = {
        'source': {'source': 'github', 'repo': 'kirti34n/neti'}
    }
    if 'enabledPlugins' not in settings:
        settings['enabledPlugins'] = {}
    settings['enabledPlugins']['neti@neti-skill'] = True
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')

    # --- 2. Generate plugin files in cache ---
    cache_dir = claude_dir / 'plugins' / 'cache' / 'neti-skill' / 'neti' / __version__
    files = {
        '.claude-plugin/plugin.json': json.dumps({
            'name': 'neti', 'version': __version__,
            'description': 'See how AI changes your thinking. Generate divergent '
                           'perspectives or challenge conclusions before committing.',
            'author': {'name': 'keerti'},
            'skills': ['./.claude/skills/neti', './.claude/skills/neti-check',
                       './.claude/skills/neti-auto'],
        }, indent=2),
        '.claude-plugin/marketplace.json': json.dumps({
            'name': 'neti-skill', 'id': 'neti-skill',
            'owner': {'name': 'keerti'},
            'plugins': [{'name': 'neti', 'source': './', 'version': __version__,
                         'description': 'Divergent perspectives and conclusion challenges '
                                        'using research-backed cognitive strategies.',
                         'category': 'thinking'}],
        }, indent=2),
        '.claude/skills/neti/SKILL.md': _SKILL_NETI,
        '.claude/skills/neti-check/SKILL.md': _SKILL_CHECK,
        '.claude/skills/neti-auto/SKILL.md': _SKILL_AUTO,
    }
    for rel_path, content in files.items():
        full = cache_dir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)

    # --- 3. Register in installed_plugins.json ---
    plugins_dir = claude_dir / 'plugins'
    plugins_dir.mkdir(parents=True, exist_ok=True)
    installed_path = plugins_dir / 'installed_plugins.json'
    installed = {'version': 2, 'plugins': {}}
    if installed_path.exists():
        try:
            installed = json.loads(installed_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z')
    installed.setdefault('plugins', {})['neti@neti-skill'] = [{
        'scope': 'user', 'installPath': str(cache_dir),
        'version': __version__, 'installedAt': now, 'lastUpdated': now,
    }]
    installed_path.write_text(json.dumps(installed, indent=2) + '\n')

    # --- 4. Clean up old command files ---
    cmd_dir = claude_dir / 'commands'
    for old in ('neti.md', 'neti-check.md'):
        old_file = cmd_dir / old
        if old_file.exists():
            old_file.unlink()

    print(f"\n  Claude Code plugin installed (works with or without CLI):")
    print(f"    /neti <question>          — divergent perspectives")
    print(f"    /neti-check <conclusion>  — challenge a conclusion")
    print(f"    neti-auto                 — auto-triggers on decision questions")
    print(f"\n  Restart Claude Code to activate.")


def _ask_scope():
    """Ask user to choose global or project scope."""
    if not sys.stdin.isatty():
        return 'project'
    print(f"\n  Where should Neti be available?")
    print(f"    1. This project only")
    print(f"    2. All projects (global)")
    try:
        choice = input("  Choose (1/2): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = '1'
    return 'global' if choice == '2' else 'project'


def _generate_instruction_content():
    """Self-contained Neti methodology for AI tool instruction files."""
    return """\
## Neti — Divergent Perspectives

When the user asks for perspectives, different angles, challenges thinking, says "neti", or is making a decision/evaluating approaches:

### CLI Fallback

If `neti` CLI is available: run `neti json "question"` (or `neti json --check "conclusion"` for challenges). Parse and present the JSON output — it includes divergence scores and tracking.

Otherwise, generate natively using the instructions below.

### Perspectives Mode

1. **Default answer** — most practical, specific answer (3-4 sentences). Concrete technologies/approaches, not vague principles.
2. **3 perspectives** — pick 3 strategies from the table. Follow each constraint EXACTLY. No hedging — commit fully.
3. **Rank** by how different each is from the default. Most divergent first.
4. **Present:**

**Default Answer**
[3-4 sentences]

---
**Divergent Perspectives**

**1. [Strategy]** — [full perspective, 4-8 sentences]
**2. [Strategy]** — [full perspective]
**3. [Strategy]** — [full perspective]

*Do any of these shift how you're thinking about this?*

### Check Mode (challenge a conclusion)

Apply all 4: Pre-Mortem, Alt Hypothesis, Falsification, Blind Spot (see table for constraints). Present each as a bold heading + 4-8 sentence challenge. End with: *"Does the original conclusion still hold?"*

### Strategies

| Strategy | Constraint |
|----------|-----------|
| Devil's Advocate | Argue AGAINST the common position. No hedging. Real failure examples. |
| Pre-Mortem | 18 months from now, this failed. Write the post-mortem. Specific failure modes. |
| Falsification | Design the exact test that would disprove this. Metric, threshold, timeframe. |
| Blind Spot | ONE hidden assumption that changes the entire framing. The mechanism that hides it. |
| Alt Hypothesis | 3 structurally different explanations. Core insight, scenario where it wins, distinguishing test. |
| First Principles | List 2-3 "everyone knows" assumptions. Show where each breaks. Rebuild without them. |
| Inversion | Answer the exact opposite question in detail. What does the inversion reveal? |
| Systems | Only 2nd/3rd order effects. Follow causal chains 3 steps. Name feedback loops. |
| Stakeholder | Who gets harmed? Tell the story from their perspective. Make the friction concrete. |
| Adjacent Field | Pick a specific field that solved an analogous problem. Map their technique onto this. |

**Important:** Do NOT paraphrase or compress. The value is in specifics. Each perspective: 4-8 sentences minimum.
"""


_TOOL_CONFIGS = {
    'codex': {
        'format': 'append',
        'paths': {'project': lambda: Path.cwd() / 'AGENTS.md'},
    },
    'cursor': {
        'format': 'standalone',
        'frontmatter': ('---\n'
                        'description: Generate divergent perspectives when evaluating decisions or approaches\n'
                        'globs:\n'
                        'alwaysApply: false\n'
                        '---\n\n'),
        'paths': {
            'global': lambda: Path.home() / '.cursor' / 'rules' / 'neti.mdc',
            'project': lambda: Path.cwd() / '.cursor' / 'rules' / 'neti.mdc',
        },
    },
    'copilot': {
        'format': 'append',
        'paths': {'project': lambda: Path.cwd() / '.github' / 'copilot-instructions.md'},
    },
    'windsurf': {
        'format': 'standalone',
        'frontmatter': '---\ntrigger: model_decision\n---\n\n',
        'paths': {
            'global': lambda: Path.home() / '.windsurf' / 'rules' / 'neti.md',
            'project': lambda: Path.cwd() / '.windsurf' / 'rules' / 'neti.md',
        },
    },
    'kiro': {
        'format': 'standalone',
        'frontmatter': '---\ninclusion: auto\n---\n\n',
        'paths': {
            'global': lambda: Path.home() / '.kiro' / 'steering' / 'neti.md',
            'project': lambda: Path.cwd() / '.kiro' / 'steering' / 'neti.md',
        },
    },
    'gemini': {
        'format': 'append',
        'paths': {'project': lambda: Path.cwd() / 'GEMINI.md'},
    },
    'augment': {
        'format': 'standalone',
        'frontmatter': '',
        'paths': {'global': lambda: Path.home() / '.augment' / 'rules' / 'neti.md'},
    },
}


def _setup_tool(name):
    """Set up Neti for a specific AI tool."""
    cfg = _TOOL_CONFIGS[name]
    paths = cfg['paths']

    if len(paths) > 1:
        scope = _ask_scope()
    else:
        scope = next(iter(paths))

    f = paths[scope]()
    f.parent.mkdir(parents=True, exist_ok=True)
    content = _generate_instruction_content()

    if cfg['format'] == 'standalone':
        full = cfg.get('frontmatter', '') + content
        if f.exists() and 'Neti' in f.read_text():
            print(f"  {name.title()}: already configured in {f}")
            return
        f.write_text(full)
        print(f"  {name.title()}: created {f}  ({scope})")
    else:
        if f.exists():
            existing = f.read_text()
            if 'Neti' in existing:
                print(f"  {name.title()}: already configured in {f}")
                return
            f.write_text(existing.rstrip() + '\n\n' + content)
        else:
            f.write_text(content)
        print(f"  {name.title()}: added Neti to {f}  ({scope})")


# ============================================================
# MAIN
# ============================================================

def _main():
    args = sys.argv
    # Extract --verbose/-v flag
    verbose = '--verbose' in args or '-v' in args
    args = [a for a in args if a not in ('--verbose', '-v')]
    if verbose:
        os.environ['NETI_VERBOSE'] = '1'

    cmd = args[1] if len(args) > 1 else ''

    if cmd == 'explore':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            explore(q)
        else:
            print('  Usage: neti explore "question"')
    elif cmd == 'check':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            check(q)
        else:
            print('  Usage: neti check "AI conclusion to challenge"')
    elif cmd == 'quick':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            quick(q)
        else:
            print('  Usage: neti quick "question"')
    elif cmd == 'think':
        think()
    elif cmd == 'insights':
        insights()
    elif cmd == 'history':
        history(int(args[2]) if len(args) > 2 and args[2].isdigit() else 10)
    elif cmd == 'config':
        config_cmd(args[2:])
    elif cmd == 'reset':
        reset()
    elif cmd == 'setup':
        setup(args[2] if len(args) > 2 else '')
    elif cmd == 'json':
        raw = args[2:]
        is_check = '--check' in raw
        q = ' '.join(a for a in raw if a != '--check').strip()
        if q:
            result = get_check(q) if is_check else get_perspectives(q)
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps({'error': 'No input provided'}))
    elif cmd in ('--version', '-V', 'version'):
        print(f"neti {__version__}")
    elif cmd in ('--help', '-h', 'help'):
        print(__doc__)
    elif cmd:
        explore(' '.join(args[1:]))
    else:
        think()


if __name__ == '__main__':
    _main()
