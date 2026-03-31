#!/usr/bin/env python3
"""
Prism — Think different. Not different answers. Different angles.

One question through multiple lenses. Measures how your thinking shifts.
Works with any LLM: OpenAI, Claude, Gemini, Ollama, or any OpenAI-compatible endpoint.

  prism.py explore "question"   # before → perspectives → after → measure
  prism.py code "describe task" # coding perspectives (edge cases, security, simplify, scale, review)
  prism.py think                # what's on your mind?
  prism.py quick "question"     # just show perspectives, no measurement
  prism.py insights             # how your thinking is changing
  prism.py history              # recent sessions
  prism.py config [key] [val]   # show/set configuration
  prism.py reset                # fresh start
  prism.py serve                # start MCP server (for Claude Code, Codex, Cursor, etc.)
"""

import json, sys, time, hashlib, random, re, math, os, threading
from pathlib import Path
from datetime import datetime

SELF = Path(__file__)
HOME = SELF.parent
STATE_FILE = HOME / "prism_state.json"
LOG_FILE = HOME / "prism.log"
VERSION = 1

# ============================================================
# EMBEDDINGS — sentence-transformers, 384D, CPU (always local)
# ============================================================

_EMBEDDER = None

def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        return _EMBEDDER
    except ImportError:
        return None

def _embed(text):
    model = _load_embedder()
    if model is None:
        return None
    return model.encode(text, normalize_embeddings=True).tolist()

def _cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    return max(0.0, 1.0 - sum(x * y for x, y in zip(a, b)))


# ============================================================
# LLM PROVIDERS — unified interface, any backend
# ============================================================

def _detect_provider():
    """Auto-detect available LLM provider."""
    import urllib.request
    # Check Ollama first (local, free)
    try:
        req = urllib.request.Request('http://localhost:11434/api/tags')
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        models = [m['name'] for m in data.get('models', [])]
        if models:
            return {'provider': 'ollama', 'model': models[0].split(':')[0] + ':' + models[0].split(':')[-1] if ':' in models[0] else models[0]}
    except:
        pass
    # Check env vars
    if os.environ.get('OPENAI_API_KEY'):
        return {'provider': 'openai', 'model': 'gpt-4o-mini'}
    if os.environ.get('ANTHROPIC_API_KEY'):
        return {'provider': 'anthropic', 'model': 'claude-sonnet-4-20250514'}
    if os.environ.get('GOOGLE_API_KEY'):
        return {'provider': 'gemini', 'model': 'gemini-2.0-flash'}
    if os.environ.get('OPENROUTER_API_KEY'):
        return {'provider': 'openrouter', 'model': 'anthropic/claude-sonnet-4-20250514'}
    return None


def _llm_call(system_prompt, user_prompt, config):
    """Unified LLM call. Routes to configured provider."""
    import urllib.request, urllib.error, socket
    provider = config.get('provider', 'ollama')
    model = config.get('model', '')
    temp = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens', 500)

    old_timeout = socket.getdefaulttimeout()

    for attempt in range(2):
        try:
            if provider == 'ollama':
                socket.setdefaulttimeout(120)
                url = config.get('endpoint', 'http://localhost:11434') + '/api/chat'
                content = user_prompt
                # Qwen3 thinking models need extra tokens for internal reasoning
                predict = max_tokens
                if 'qwen' in model.lower():
                    predict = max(max_tokens * 3, 1500)
                body = json.dumps({
                    'model': model,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': content},
                    ],
                    'stream': False,
                    'options': {'temperature': temp, 'num_predict': predict}
                }).encode()
                req = urllib.request.Request(url, data=body,
                    headers={'Content-Type': 'application/json'})
                resp = urllib.request.urlopen(req, timeout=120)
                raw = _read_with_timeout(resp, timeout=120)
                result = json.loads(raw)
                text = result.get('message', {}).get('content', '')
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                socket.setdefaulttimeout(old_timeout)
                return text

            elif provider == 'openai' or provider == 'openrouter' or provider == 'custom':
                socket.setdefaulttimeout(60)
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
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    'temperature': temp,
                    'max_tokens': max_tokens,
                }).encode()
                headers = {'Content-Type': 'application/json'}
                if key:
                    headers['Authorization'] = f'Bearer {key}'
                req = urllib.request.Request(url, data=body, headers=headers)
                resp = urllib.request.urlopen(req, timeout=60)
                raw = _read_with_timeout(resp, timeout=60)
                result = json.loads(raw)
                choices = result.get('choices', [])
                text = choices[0].get('message', {}).get('content', '') if choices else ''
                socket.setdefaulttimeout(old_timeout)
                return text

            elif provider == 'anthropic':
                socket.setdefaulttimeout(60)
                key = os.environ.get('ANTHROPIC_API_KEY', '')
                body = json.dumps({
                    'model': model,
                    'max_tokens': max_tokens,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                    'temperature': temp,
                }).encode()
                req = urllib.request.Request(
                    'https://api.anthropic.com/v1/messages',
                    data=body,
                    headers={
                        'Content-Type': 'application/json',
                        'x-api-key': key,
                        'anthropic-version': '2023-06-01',
                    })
                resp = urllib.request.urlopen(req, timeout=60)
                raw = _read_with_timeout(resp, timeout=60)
                result = json.loads(raw)
                blocks = result.get('content', [])
                text = ' '.join(b.get('text', '') for b in blocks if b.get('type') == 'text')
                socket.setdefaulttimeout(old_timeout)
                return text

            elif provider == 'gemini':
                socket.setdefaulttimeout(90)
                key = os.environ.get('GOOGLE_API_KEY', '')
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                body = json.dumps({
                    'system_instruction': {'parts': [{'text': system_prompt}]},
                    'contents': [{'parts': [{'text': user_prompt}]}],
                    'generationConfig': {'temperature': temp, 'maxOutputTokens': max_tokens}
                }).encode()
                req = urllib.request.Request(url, data=body,
                    headers={'Content-Type': 'application/json'})
                resp = urllib.request.urlopen(req, timeout=90)
                raw = _read_with_timeout(resp, timeout=90)
                result = json.loads(raw)
                parts = result.get('candidates', [{}])[0].get('content', {}).get('parts', [])
                text = parts[0].get('text', '') if parts else ''
                socket.setdefaulttimeout(old_timeout)
                return text

        except Exception as e:
            socket.setdefaulttimeout(old_timeout)
            if attempt == 0:
                time.sleep(1)
            else:
                return ''
    socket.setdefaulttimeout(old_timeout)
    return ''


def _read_with_timeout(resp, timeout=60):
    result = [None]
    def _read():
        try:
            result[0] = resp.read()
        except:
            result[0] = b''
    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return b''
    return result[0] if result[0] else b''


# ============================================================
# PERSPECTIVE STRATEGIES — structural constraints, not roles
# ============================================================

STRATEGIES = {
    'default': {
        'name': 'Default',
        'system': 'Answer the question directly and honestly.',
    },
    'devils_advocate': {
        'name': "Devil's Advocate",
        'system': 'You MUST argue against the most likely answer to this question. Do not present both sides. Do not hedge. Take the opposing position and defend it with your strongest arguments. Be specific and committed.',
        'prefix': 'The common answer is probably obvious. Argue against it:\n\n',
    },
    'blind_spot': {
        'name': 'Blind Spot',
        'system': 'Your job is to identify exactly ONE thing that most people overlook or get wrong about this topic. Do not give a balanced overview. Name the blind spot, explain why people miss it, and explain what changes when you see it.',
        'prefix': 'What is the one thing most people miss about this:\n\n',
    },
    'first_principles': {
        'name': 'First Principles',
        'system': 'Strip this question to its most basic assumptions. List the 2-3 assumptions everyone takes for granted. Then question each one. What if the foundational assumption is wrong? Rebuild the answer from scratch.',
        'prefix': 'Break this down to first principles:\n\n',
    },
    'temporal': {
        'name': 'Temporal',
        'system': 'Answer this question three times: how someone would have answered 50 years ago, how someone answers today, and how someone will answer 50 years from now. Focus on what CHANGES between timeframes and why.',
        'prefix': 'How does the answer change across time:\n\n',
    },
    'stakeholder': {
        'name': 'Stakeholder',
        'system': 'Identify the person or group most HARMED by the conventional answer. Explain entirely from their perspective. Do not balance with other perspectives. Give voice only to the one who loses.',
        'prefix': 'Who loses when we answer this the standard way:\n\n',
    },
    'systems': {
        'name': 'Systems',
        'system': 'Ignore the direct answer. Focus entirely on second-order and third-order effects. What happens BECAUSE of the obvious answer? Follow the chain at least three steps. Indirect consequences matter more than the direct answer.',
        'prefix': 'What are the downstream consequences:\n\n',
    },
    'constraint_removal': {
        'name': 'Constraint Removal',
        'system': 'Identify the single biggest constraint shaping how people think about this (money, time, norms, physics, law). Now remove it. How does the answer change completely? Be specific about what becomes possible.',
        'prefix': 'What is the biggest constraint, and what happens without it:\n\n',
    },
    'emotional': {
        'name': 'Emotional',
        'system': 'Do not analyze intellectually. Describe the emotional reality of this question. What does it feel like to face this? What fears, hopes, anxieties, or desires actually drive people\'s answers? Name the emotions, not the logic.',
        'prefix': 'What is the emotional truth:\n\n',
    },
    'inversion': {
        'name': 'Inversion',
        'system': 'Answer the exact OPPOSITE of this question. If it asks how to succeed, explain how to guarantee failure. If it asks what matters, explain what is irrelevant. Be thorough about the inverted answer. Let the contrast reveal what the direct answer misses.',
        'prefix': 'Answer the opposite:\n\n',
    },
    # --- Coding-specific strategies ---
    'edge_cases': {
        'name': 'Edge Cases',
        'system': 'You are a QA engineer who only thinks about what breaks. List the 3-5 most dangerous edge cases, race conditions, or unexpected inputs this code or approach will encounter. For each one, explain exactly how it fails and what the symptom looks like in production.',
        'prefix': 'What are the edge cases and failure modes:\n\n',
        'tags': ['code'],
    },
    'security': {
        'name': 'Security',
        'system': 'You are a security auditor. Identify the attack surface. What can be injected, leaked, or escalated? Name specific vulnerability classes (OWASP top 10, supply chain, timing attacks, etc.). Do not give general advice — name the specific risk in THIS code/approach.',
        'prefix': 'What are the security implications:\n\n',
        'tags': ['code'],
    },
    'simplify': {
        'name': 'Simplify',
        'system': 'This approach is overengineered. Find a solution that is at least 50% simpler. Fewer files, fewer abstractions, fewer dependencies. Show the simpler version. If the simple version has tradeoffs, name them, but default to simple.',
        'prefix': 'What is a dramatically simpler approach:\n\n',
        'tags': ['code'],
    },
    'scale': {
        'name': 'Scale',
        'system': 'This works now. What breaks at 10x users? 100x? 1000x? Identify the specific bottleneck that hits first. Name the component, the metric that degrades, and the order of magnitude where it fails. Then suggest the minimum change to push the bottleneck one order of magnitude further.',
        'prefix': 'What breaks when this scales:\n\n',
        'tags': ['code'],
    },
    'reviewer': {
        'name': 'Code Review',
        'system': 'Review this as a senior engineer who has seen this pattern fail before. What would you flag in a PR review? Focus on: maintainability, naming, error handling, testability, and hidden coupling. Be specific — quote the problematic pattern, explain why it will cause pain later.',
        'prefix': 'Give a tough but fair code review:\n\n',
        'tags': ['code'],
    },
    'alternative_stack': {
        'name': 'Alt Stack',
        'system': 'Solve the same problem using a completely different technology stack, language, or paradigm. If the original uses an ORM, try raw SQL. If it uses microservices, try a monolith. If it uses REST, try GraphQL or RPC. Show enough of the alternative to evaluate the tradeoff.',
        'prefix': 'Solve this with a completely different approach:\n\n',
        'tags': ['code'],
    },
}

# Strategy tags for mode-specific selection
CODE_STRATEGIES = [k for k, v in STRATEGIES.items() if 'code' in v.get('tags', [])]
THINKING_STRATEGIES = [k for k, v in STRATEGIES.items() if k != 'default' and 'code' not in v.get('tags', [])]


def _select_strategies(state, n=4, mode='think'):
    """Pick n strategies: top 2 by weight + 2 random for exploration.
    mode='think' for general thinking, 'code' for coding perspectives, 'all' for mixed."""
    weights = state.get('strategy_weights', {})

    if mode == 'code':
        pool = CODE_STRATEGIES
    elif mode == 'think':
        pool = THINKING_STRATEGIES
    else:
        pool = [k for k in STRATEGIES if k != 'default']

    # Sort by weight (higher = more useful historically)
    scored = [(k, weights.get(k, 1.0)) for k in pool]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Top 2 by weight
    top = [s[0] for s in scored[:min(2, len(scored))]]

    # Remaining random for exploration
    remaining = [s[0] for s in scored if s[0] not in top]
    n_explore = min(n - len(top), len(remaining))
    explore_picks = random.sample(remaining, n_explore) if n_explore > 0 else []

    return (top + explore_picks)[:n]


def _generate_perspectives(question, strategies, config, quiet=False):
    """Generate default + N perspectives. Parallel for APIs, sequential for Ollama."""
    provider = config.get('provider', 'ollama')
    results = {}
    _p = (lambda *a, **kw: None) if quiet else (lambda *a, **kw: print(*a, **kw))

    # Always generate default first
    _p(f"  [1/{len(strategies)+1}] Default", end="", flush=True)
    default_resp = _llm_call(
        STRATEGIES['default']['system'],
        question,
        {**config, 'max_tokens': config.get('max_tokens_default', 600)}
    )
    results['default'] = default_resp
    _p(f" — done" if default_resp else f" — failed")

    # Generate perspectives
    if provider in ('openai', 'anthropic', 'gemini', 'openrouter', 'custom'):
        # Parallel for API providers
        lock = threading.Lock()
        def _gen(idx, key):
            s = STRATEGIES[key]
            prompt = s.get('prefix', '') + question
            resp = _llm_call(s['system'], prompt,
                {**config, 'max_tokens': config.get('max_tokens_perspective', 400)})
            with lock:
                results[key] = resp
            _p(f"  [{idx+2}/{len(strategies)+1}] {s['name']} — {'done' if resp else 'failed'}")

        threads = []
        for i, key in enumerate(strategies):
            t = threading.Thread(target=_gen, args=(i, key), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=120)
    else:
        # Sequential for Ollama
        for i, key in enumerate(strategies):
            s = STRATEGIES[key]
            _p(f"  [{i+2}/{len(strategies)+1}] {s['name']}", end="", flush=True)
            prompt = s.get('prefix', '') + question
            resp = _llm_call(s['system'], prompt,
                {**config, 'max_tokens': config.get('max_tokens_perspective', 400)})
            results[key] = resp
            _p(f" — {'done' if resp else 'failed'}")

    return results


# ============================================================
# STATE
# ============================================================

def _default_config():
    detected = _detect_provider()
    if detected:
        return {
            'provider': detected['provider'],
            'model': detected['model'],
            'temperature': 0.7,
            'max_tokens_default': 600,
            'max_tokens_perspective': 400,
            'num_perspectives': 4,
            'num_shown': 3,
        }
    return {
        'provider': 'ollama',
        'model': 'qwen3:8b',
        'temperature': 0.7,
        'max_tokens_default': 600,
        'max_tokens_perspective': 400,
        'num_perspectives': 4,
        'num_shown': 3,
    }


def new_state():
    return {
        'version': VERSION,
        'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:12],
        'created': datetime.now().isoformat(),
        'config': _default_config(),
        'strategy_weights': {k: 1.0 for k in STRATEGIES if k != 'default'},
        'sessions': [],
    }

def load():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if data.get('version') == VERSION:
                return data
        except:
            pass
    return new_state()

def save(state):
    state['last_used'] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))

def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# ============================================================
# MEASUREMENT
# ============================================================

def _measure_shift(emb_before, emb_after):
    """How far did the human's thinking move?"""
    if emb_before is None or emb_after is None:
        return None
    d = _cosine_distance(emb_before, emb_after)
    if d < 0.05:
        return {'shift': d, 'label': 'none'}
    elif d < 0.15:
        return {'shift': d, 'label': 'minor'}
    elif d < 0.35:
        return {'shift': d, 'label': 'moderate'}
    elif d < 0.55:
        return {'shift': d, 'label': 'large'}
    else:
        return {'shift': d, 'label': 'paradigm'}


def _measure_direction(emb_before, emb_after, perspective_embs):
    """Where did the human move? Toward a perspective, or independent?"""
    if emb_before is None or emb_after is None:
        return 'unknown'

    before_dists = {k: _cosine_distance(emb_before, e) for k, e in perspective_embs.items() if e}
    after_dists = {k: _cosine_distance(emb_after, e) for k, e in perspective_embs.items() if e}

    if not before_dists or not after_dists:
        return 'unknown'

    before_nearest = min(before_dists.values())
    after_nearest_key = min(after_dists, key=after_dists.get)
    after_nearest = after_dists[after_nearest_key]

    if after_nearest < before_nearest - 0.05:
        return f'toward_{after_nearest_key}'
    elif after_nearest > before_nearest + 0.05:
        return 'independent'
    else:
        return 'stable'


def _measure_independence(emb_after, perspective_embs):
    """How far is human from nearest AI response? Higher = more independent."""
    if emb_after is None:
        return None
    dists = [_cosine_distance(emb_after, e) for e in perspective_embs.values() if e]
    if not dists:
        return None
    nearest = min(dists)
    return min(1.0, nearest / 0.6)


# ============================================================
# FEEDBACK
# ============================================================

def _update_weights(state, shown_strategies, rated_strategy, shift_val, direction):
    """Update strategy weights based on explicit and implicit signals."""
    w = state.get('strategy_weights', {})

    for key in shown_strategies:
        # Slight boost for being divergent enough to be shown
        w[key] = w.get(key, 1.0) * 1.03

    if rated_strategy and rated_strategy in w:
        # Strong boost for being rated useful
        w[rated_strategy] = w[rated_strategy] * 1.2
        # Slight decay for shown-but-not-rated
        for key in shown_strategies:
            if key != rated_strategy:
                w[key] = w.get(key, 1.0) * 0.95

    # Implicit: shift toward a perspective boosts that perspective
    if direction and direction.startswith('toward_'):
        target = direction.replace('toward_', '')
        if target in w:
            w[target] = w.get(target, 1.0) * 1.1

    # No shift = nothing worked
    if shift_val is not None and shift_val < 0.05:
        for key in shown_strategies:
            w[key] = w.get(key, 1.0) * 0.97

    # Decay all toward 1.0 (prevents runaway)
    for key in w:
        w[key] = w[key] * 0.99 + 1.0 * 0.01

    state['strategy_weights'] = w


# ============================================================
# COMMANDS
# ============================================================

PROMPTS = [
    "What are you avoiding right now?",
    "What do you believe that you can't prove?",
    "Where are you stuck?",
    "What would you do if you knew it would fail?",
    "What have you been wrong about recently?",
    "What decision are you putting off?",
    "What pattern are you in right now?",
    "What would change if you stopped?",
    "What are you pretending isn't a problem?",
    "What are you optimizing for, and should you be?",
    "What would you tell yourself from a year ago?",
    "What do you know that nobody asked you about?",
    "What's the thing you keep coming back to?",
    "What are you building and why?",
    "What would you do differently if nobody was watching?",
]


def explore(question, mode='think'):
    """The full experiment: before → perspectives → after → measure.
    mode='think' for general, 'code' for coding perspectives."""
    state = load()
    cfg = state.get('config', _default_config())

    mode_label = "Code Review" if mode == 'code' else "Explore"
    print(f"\n  PRISM — {mode_label}")
    print(f"  {'='*56}")
    print(f"  {question}")
    print(f"  {'='*56}\n")

    # Step 1: Human's initial take
    prompt_text = "Your approach (before seeing perspectives):" if mode == 'code' else "Your take (before seeing anything):"
    print(f"  {prompt_text}", flush=True)
    human_before = None
    if sys.stdin.isatty():
        try:
            human_before = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            pass
    if not human_before:
        human_before = None

    # Step 2: Generate perspectives
    print(f"\n  Generating perspectives ({cfg.get('provider', '?')}/{cfg.get('model', '?')})...", flush=True)
    strategies = _select_strategies(state, cfg.get('num_perspectives', 4), mode=mode)
    responses = _generate_perspectives(question, strategies, cfg)

    # Filter empty responses
    responses = {k: v for k, v in responses.items() if v}
    if 'default' not in responses:
        print("  Default response failed. Check your LLM configuration.")
        print(f"  Provider: {cfg.get('provider')} | Model: {cfg.get('model')}")
        return

    non_default = {k: v for k, v in responses.items() if k != 'default'}
    if not non_default:
        print("  No perspectives generated. Showing default only.\n")
        print(f"  [Default]:")
        _print_wrapped(responses['default'], indent=4)
        return

    # Step 3: Embed and select most divergent
    has_embedder = _load_embedder() is not None
    shown_keys = list(non_default.keys())
    divergences = {}

    if has_embedder:
        print(f"  Selecting most divergent...", flush=True)
        default_emb = _embed(responses['default'])
        perspective_embs = {}
        for key, text in non_default.items():
            emb = _embed(text)
            perspective_embs[key] = emb
            if emb and default_emb:
                divergences[key] = _cosine_distance(default_emb, emb)

        # Sort by divergence, pick top N
        num_shown = cfg.get('num_shown', 3)
        sorted_keys = sorted(divergences, key=divergences.get, reverse=True)
        shown_keys = sorted_keys[:num_shown]
    else:
        default_emb = None
        perspective_embs = {k: None for k in non_default}

    # Step 4: Show the collision
    print(f"\n  {'─'*56}")
    print(f"  DEFAULT ANSWER")
    print(f"  {'─'*56}")
    _print_wrapped(responses['default'], indent=4)

    for i, key in enumerate(shown_keys):
        s = STRATEGIES.get(key, {})
        div_str = f" (distance: {divergences[key]:.3f})" if key in divergences else ""
        print(f"\n  {'─'*56}")
        print(f"  PERSPECTIVE {i+1}: {s.get('name', key)}{div_str}")
        print(f"  {'─'*56}")
        _print_wrapped(responses[key], indent=4)

    # Step 5: Human's revised thinking
    print(f"\n  {'─'*56}")
    print(f"  Now what do you think?", flush=True)
    human_after = None
    if sys.stdin.isatty():
        try:
            human_after = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            pass
    if not human_after:
        human_after = None

    # Step 6: Measure
    emb_before = _embed(human_before) if human_before else None
    emb_after = _embed(human_after) if human_after else None

    # Build perspective embeddings map including default
    all_embs = {'default': default_emb if has_embedder else None}
    all_embs.update(perspective_embs)

    shift_data = _measure_shift(emb_before, emb_after)
    direction = _measure_direction(emb_before, emb_after, all_embs)
    independence = _measure_independence(emb_after, all_embs)

    # Build session entry
    session = {
        'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:8],
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'human_before': human_before[:500] if human_before else None,
        'human_after': human_after[:500] if human_after else None,
        'strategies_generated': ['default'] + strategies,
        'strategies_shown': shown_keys,
        'divergences': {k: round(v, 4) for k, v in divergences.items()},
        'shift': round(shift_data['shift'], 4) if shift_data else None,
        'shift_label': shift_data['label'] if shift_data else None,
        'direction': direction,
        'independence': round(independence, 4) if independence is not None else None,
        'convergence_score': round(_cosine_distance(emb_after, all_embs.get('default')), 4) if emb_after and all_embs.get('default') else None,
        'user_rating': None,
    }

    # Show measurement
    if shift_data:
        print(f"\n  {'='*56}")
        print(f"  MEASUREMENT")
        print(f"  {'='*56}")
        print(f"\n  Your shift: {shift_data['shift']:.4f} ({shift_data['label']})")
        if direction != 'unknown':
            label = direction.replace('toward_', 'toward ').replace('_', ' ')
            if direction.startswith('toward_'):
                strategy_name = STRATEGIES.get(direction.replace('toward_', ''), {}).get('name', direction)
                label = f"toward {strategy_name}"
            print(f"  Direction:  {label}")
        if independence is not None:
            print(f"  Independence: {independence:.0%}")

    # Step 7: Feedback (one keystroke)
    rated = None
    if sys.stdin.isatty() and shown_keys:
        try:
            prompt_parts = [f"{i+1}={STRATEGIES.get(k, {}).get('name', k)}" for i, k in enumerate(shown_keys)]
            print(f"\n  Most useful? ({', '.join(prompt_parts)}, or Enter to skip)")
            choice = input("  > ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(shown_keys):
                rated = shown_keys[int(choice) - 1]
                session['user_rating'] = rated
        except (EOFError, KeyboardInterrupt):
            pass

    # Update weights
    _update_weights(
        state, shown_keys, rated,
        shift_data['shift'] if shift_data else None,
        direction
    )

    # Save session
    state.setdefault('sessions', []).append(session)
    save(state)

    n = len(state['sessions'])
    if n < 5:
        print(f"\n  Session logged ({n}/5 for first insights).\n")
    else:
        print(f"\n  Session logged. Run 'prism insights' for patterns.\n")

    log(f"explore: q='{question[:50]}' shift={session.get('shift', '?')} dir={direction}")


def quick(question):
    """Show divergent perspectives without before/after measurement."""
    state = load()
    cfg = state.get('config', _default_config())

    print(f"\n  PRISM — Quick View")
    print(f"  {question}\n")

    strategies = _select_strategies(state, cfg.get('num_perspectives', 4))
    print(f"  Generating ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    responses = _generate_perspectives(question, strategies, cfg)
    responses = {k: v for k, v in responses.items() if v}

    if 'default' not in responses:
        print("  Failed. Check LLM configuration.")
        return

    # Embed and sort by divergence
    has_embedder = _load_embedder() is not None
    if has_embedder:
        default_emb = _embed(responses['default'])
        divergences = {}
        for key in responses:
            if key != 'default':
                emb = _embed(responses[key])
                if emb and default_emb:
                    divergences[key] = _cosine_distance(default_emb, emb)
        sorted_keys = sorted(divergences, key=divergences.get, reverse=True)
    else:
        sorted_keys = [k for k in responses if k != 'default']
        divergences = {}

    print(f"\n  {'─'*56}")
    print(f"  DEFAULT")
    print(f"  {'─'*56}")
    _print_wrapped(responses['default'], indent=4)

    for key in sorted_keys[:cfg.get('num_shown', 3)]:
        s = STRATEGIES.get(key, {})
        div = f" ({divergences[key]:.3f})" if key in divergences else ""
        print(f"\n  {'─'*56}")
        print(f"  {s.get('name', key).upper()}{div}")
        print(f"  {'─'*56}")
        _print_wrapped(responses[key], indent=4)

    print(f"\n  What do you think?\n")
    log(f"quick: q='{question[:50]}'")


def think():
    """What's on your mind? Or get a prompt."""
    print(f"\n  What's on your mind? (or Enter for a prompt)", flush=True)
    question = None
    if sys.stdin.isatty():
        try:
            question = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            pass
    if not question:
        question = random.choice(PROMPTS)
        print(f"  → {question}")
    explore(question)


def insights():
    """Show thinking patterns over time."""
    state = load()
    sessions = state.get('sessions', [])

    print(f"\n  PRISM — Insights")
    print(f"  {'─'*40}")
    print(f"  Sessions: {len(sessions)}")

    if len(sessions) < 3:
        print(f"\n  Need at least 3 sessions for insights.")
        print(f"  Run: prism explore \"your question\"\n")
        return

    # Shift stats
    shifts = [s['shift'] for s in sessions if s.get('shift') is not None]
    if shifts:
        print(f"\n  Shift stats:")
        print(f"    Average: {sum(shifts)/len(shifts):.4f}")
        print(f"    Largest: {max(shifts):.4f}")
        labels = [s.get('shift_label', '?') for s in sessions if s.get('shift_label')]
        from collections import Counter
        lc = Counter(labels)
        for label, count in lc.most_common():
            print(f"    {label:>10}: {count}")

    # Independence
    indeps = [s['independence'] for s in sessions if s.get('independence') is not None]
    if indeps:
        print(f"\n  Independence: {sum(indeps)/len(indeps):.0%} average")

    # Strategy effectiveness
    if len(sessions) >= 5:
        strat_shifts = {}
        for s in sessions:
            if s.get('shift') is not None:
                for key in s.get('strategies_shown', []):
                    strat_shifts.setdefault(key, []).append(s['shift'])

        if strat_shifts:
            print(f"\n  What moves you:")
            ranked = sorted(strat_shifts.items(),
                key=lambda x: sum(x[1])/len(x[1]), reverse=True)
            for key, shifts_list in ranked[:5]:
                avg = sum(shifts_list) / len(shifts_list)
                name = STRATEGIES.get(key, {}).get('name', key)
                bar = '#' * int(avg * 30)
                print(f"    {name:>20}: {avg:.3f} |{bar}|")

    # Direction patterns
    directions = [s.get('direction', '') for s in sessions if s.get('direction')]
    if directions:
        from collections import Counter
        dc = Counter(directions)
        print(f"\n  Direction:")
        for d, count in dc.most_common():
            label = d.replace('toward_', '→ ').replace('_', ' ')
            if d.startswith('toward_'):
                sname = STRATEGIES.get(d.replace('toward_', ''), {}).get('name', d)
                label = f"→ {sname}"
            print(f"    {label:>25}: {count}")

    # Convergence tracking (20+ sessions)
    conv_scores = [s['convergence_score'] for s in sessions if s.get('convergence_score') is not None]
    if len(conv_scores) >= 10:
        # Linear regression
        n = len(conv_scores)
        recent = conv_scores[-20:]  # last 20
        nr = len(recent)
        x_mean = (nr - 1) / 2
        y_mean = sum(recent) / nr
        num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(nr))
        slope = num / den if den > 0 else 0

        print(f"\n  Default convergence (last {nr} sessions):")
        if slope < -0.01:
            print(f"    CONVERGING (slope: {slope:.4f})")
            print(f"    Your thinking is moving closer to AI defaults.")
            print(f"    The sycophancy attractor may be pulling you in.")
        elif slope > 0.01:
            print(f"    DIVERGING (slope: {slope:.4f})")
            print(f"    You're becoming MORE independent over time.")
        else:
            print(f"    STABLE (slope: {slope:.4f})")
            print(f"    Your independence is holding steady.")

    print()


def history():
    """Show recent sessions."""
    state = load()
    sessions = state.get('sessions', [])

    print(f"\n  PRISM — History ({len(sessions)} sessions)")
    print(f"  {'─'*56}")

    if not sessions:
        print(f"  No sessions yet. Run: prism explore \"your question\"\n")
        return

    for s in sessions[-10:]:
        shift = s.get('shift')
        shift_str = f"{shift:.3f}" if shift is not None else "  -  "
        label = (s.get('shift_label') or '')[:6]
        direction = s.get('direction', '')
        if direction.startswith('toward_'):
            sname = STRATEGIES.get(direction.replace('toward_', ''), {}).get('name', '?')
            dir_str = f"→{sname[:12]}"
        else:
            dir_str = direction[:12]
        print(f"  [{shift_str}] {label:>8} {dir_str:>14} | {s['question'][:35]}")

    print()


def config_cmd(args):
    """Show or set configuration."""
    state = load()
    cfg = state.get('config', _default_config())

    if not args:
        print(f"\n  PRISM — Configuration")
        print(f"  {'─'*40}")
        for k, v in cfg.items():
            print(f"  {k:>25}: {v}")
        print(f"\n  Set with: prism config <key> <value>")
        print(f"  Example:  prism config provider openai")
        print(f"            prism config model gpt-4o\n")
        return

    key = args[0]
    if len(args) < 2:
        print(f"  {key}: {cfg.get(key, '(not set)')}")
        return

    val = ' '.join(args[1:])
    # Type coerce
    if key in ('temperature',):
        val = float(val)
    elif key in ('max_tokens_default', 'max_tokens_perspective', 'num_perspectives', 'num_shown'):
        val = int(val)

    cfg[key] = val
    state['config'] = cfg
    save(state)
    print(f"  {key} = {val}")


def reset():
    """Fresh start."""
    if sys.stdin.isatty():
        try:
            confirm = input("  Delete all history? (yes/no): ").strip()
            if confirm.lower() != 'yes':
                print("  Cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            return
    state = new_state()
    save(state)
    print(f"\n  Prism reset. ID: {state['id']}")
    print(f"  Provider: {state['config']['provider']} | Model: {state['config']['model']}\n")


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
            current = prefix
            for word in words:
                if len(current) + len(word) + 1 > width:
                    print(current)
                    current = prefix + word
                else:
                    current = current + ' ' + word if current != prefix else current + word
            if current.strip():
                print(current)


# ============================================================
# PROGRAMMATIC API — for coding tools, MCP, hooks
# ============================================================

def get_perspectives(question, mode='think', n=4):
    """Generate perspectives and return as structured data. For tool integration."""
    state = load()
    cfg = state.get('config', _default_config())
    strategies = _select_strategies(state, n, mode=mode)
    responses = _generate_perspectives(question, strategies, cfg, quiet=True)
    responses = {k: v for k, v in responses.items() if v}

    if 'default' not in responses:
        return {'error': 'LLM call failed', 'config': cfg}

    # Embed and rank by divergence
    result = {'question': question, 'default': responses['default'], 'perspectives': []}
    has_embedder = _load_embedder() is not None
    if has_embedder:
        default_emb = _embed(responses['default'])
        for key in strategies:
            if key in responses and key != 'default':
                emb = _embed(responses[key])
                div = _cosine_distance(default_emb, emb) if emb and default_emb else 0
                result['perspectives'].append({
                    'strategy': key,
                    'name': STRATEGIES.get(key, {}).get('name', key),
                    'text': responses[key],
                    'divergence': round(div, 4),
                })
        result['perspectives'].sort(key=lambda x: x['divergence'], reverse=True)
    else:
        for key in strategies:
            if key in responses:
                result['perspectives'].append({
                    'strategy': key,
                    'name': STRATEGIES.get(key, {}).get('name', key),
                    'text': responses[key],
                })

    return result


def serve_mcp():
    """Start MCP server for integration with Claude Code, Codex, Cursor, etc."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        print("  MCP server requires fastmcp: pip install fastmcp")
        return

    mcp = FastMCP(name="Prism", instructions="Prism provides alternative perspectives on any question or code. Use it to challenge assumptions and find blind spots.")

    @mcp.tool
    def perspective(topic: str, mode: str = "think") -> str:
        """Get divergent perspectives on a topic. mode='think' for general, 'code' for coding."""
        result = get_perspectives(topic, mode=mode, n=3)
        if 'error' in result:
            return f"Error: {result['error']}"
        output = f"DEFAULT:\n{result['default']}\n\n"
        for p in result.get('perspectives', [])[:3]:
            output += f"--- {p['name']} (divergence: {p.get('divergence', '?')}) ---\n{p['text']}\n\n"
        return output

    @mcp.tool
    def code_review(description: str) -> str:
        """Get coding perspectives: edge cases, security, simplify, scale, review."""
        result = get_perspectives(description, mode='code', n=4)
        if 'error' in result:
            return f"Error: {result['error']}"
        output = f"DEFAULT APPROACH:\n{result['default']}\n\n"
        for p in result.get('perspectives', [])[:4]:
            output += f"--- {p['name']} (divergence: {p.get('divergence', '?')}) ---\n{p['text']}\n\n"
        return output

    @mcp.tool
    def challenge(assumption: str) -> str:
        """Challenge a specific assumption using devil's advocate and inversion."""
        state = load()
        cfg = state.get('config', _default_config())
        responses = _generate_perspectives(assumption, ['devils_advocate', 'inversion', 'first_principles'], cfg)
        responses = {k: v for k, v in responses.items() if v}
        output = ""
        for key, text in responses.items():
            if key != 'default':
                name = STRATEGIES.get(key, {}).get('name', key)
                output += f"--- {name} ---\n{text}\n\n"
            else:
                output = f"DEFAULT:\n{text}\n\n" + output
        return output if output else "Failed to generate perspectives."

    print("  Starting Prism MCP server...")
    print("  Register with Claude Code: claude mcp add prism -- python3 prism.py serve")
    print("  Register with Codex: add to ~/.codex/config.toml")
    mcp.run(transport='stdio')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    args = sys.argv
    cmd = args[1] if len(args) > 1 else 'think'

    if cmd == 'explore':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            explore(q)
        else:
            print("  Usage: prism explore \"your question\"")
    elif cmd == 'code':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            explore(q, mode='code')
        else:
            print("  Usage: prism code \"describe your task or paste code\"")
    elif cmd == 'think':
        think()
    elif cmd == 'quick':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            quick(q)
        else:
            print("  Usage: prism quick \"your question\"")
    elif cmd == 'insights':
        insights()
    elif cmd == 'history':
        history()
    elif cmd == 'config':
        config_cmd(args[2:])
    elif cmd == 'reset':
        reset()
    elif cmd == 'serve':
        serve_mcp()
    elif cmd == 'json':
        # Machine-readable output for hooks/integrations
        q = ' '.join(args[2:]) if len(args) > 2 else None
        mode = 'code' if '--code' in args else 'think'
        if q:
            result = get_perspectives(q.replace('--code', '').strip(), mode=mode)
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps({'error': 'No question provided'}))
    else:
        print(__doc__)
