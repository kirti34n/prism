#!/usr/bin/env python3
"""
Prism — Think different. Not different answers. Different angles.

One question through multiple lenses. Measures how your thinking shifts.
Research-backed structural constraints. Zero dependencies.

  prism.py "question"              # explore: position → perspectives → revised position
  prism.py check "AI conclusion"   # challenge an AI conclusion before committing
  prism.py quick "question"        # just show perspectives, no measurement
  prism.py think                   # random prompt → explore
  prism.py insights                # your thinking patterns over time
  prism.py history                 # recent sessions
  prism.py config [key] [val]      # show or set configuration
  prism.py setup install            # make 'prism' a global command
  prism.py setup claude            # + Claude Code slash commands
  prism.py setup all               # install + all integrations
  prism.py json "question"         # machine-readable output
  prism.py json --check "concl"    # machine-readable check output
  prism.py reset                   # fresh start

Config:  .prism.json (project) → ~/.config/prism/config.json (global) → auto-detect
"""

import json, sys, time, hashlib, random, re, math, os, threading
from pathlib import Path
from datetime import datetime
from collections import Counter

VERSION = 2

# ============================================================
# PATHS
# ============================================================

CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'prism'
STATE_FILE = CONFIG_DIR / 'state.json'
LOG_FILE = CONFIG_DIR / 'prism.log'


# ============================================================
# CONFIG
# ============================================================

def _find_project_config():
    d = Path.cwd()
    while True:
        f = d / '.prism.json'
        if f.exists():
            try:
                return json.loads(f.read_text())
            except:
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
        except:
            pass
    return {}


def _save_global_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / 'config.json').write_text(json.dumps(cfg, indent=2))


def _detect_provider():
    import urllib.request
    try:
        req = urllib.request.Request('http://localhost:11434/api/tags')
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        models = [m['name'] for m in data.get('models', [])]
        if models:
            return {'provider': 'ollama', 'model': models[0]}
    except:
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
        'provider': 'ollama', 'model': 'qwen3:8b', 'temperature': 0.7,
        'max_tokens': 300, 'strategies': 'auto',
        'num_perspectives': 4, 'num_shown': 3,
    }
    detected = _detect_provider()
    if detected:
        defaults['provider'] = detected['provider']
        defaults['model'] = detected['model']
    return {**defaults, **_load_global_config(), **_find_project_config()}


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


def _cosine_distance_emb(a, b):
    return max(0.0, 1.0 - sum(x * y for x, y in zip(a, b)))


def _tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def _bow_distance(a, b):
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


def _distance(a, b):
    ea, eb = _embed(a), _embed(b)
    if ea is not None and eb is not None:
        return _cosine_distance_emb(ea, eb)
    return _bow_distance(a, b)


def _measurement_method():
    return 'semantic' if _load_embedder() else 'lexical'


def _measure_direction(before, after, responses):
    if not before or not after:
        return 'unknown'
    bd = {k: _distance(before, v) for k, v in responses.items() if v}
    ad = {k: _distance(after, v) for k, v in responses.items() if v}
    if not bd or not ad:
        return 'unknown'
    bn = min(bd.values())
    ank = min(ad, key=ad.get)
    an = ad[ank]
    if an < bn - 0.05:
        return f'toward_{ank}'
    elif an > bn + 0.05:
        return 'independent'
    return 'stable'


def _measure_independence(after, responses):
    if not after:
        return None
    dists = [_distance(after, v) for v in responses.values() if v]
    return min(1.0, min(dists) / 0.6) if dists else None


def _classify_session(conf_before, conf_after, shift, direction, after_text, question):
    """Classify session based on confidence + text + direction signals."""
    # Reframing: user asked a different question
    is_reframing = (after_text and '?' in after_text
                    and _distance(question, after_text) > 0.3)
    if is_reframing:
        return 'reframing'

    # Destabilization: confidence dropped significantly
    if conf_before is not None and conf_after is not None:
        if conf_after - conf_before <= -3:
            return 'destabilization'

    # Adoption: moved toward a model response
    if direction and direction.startswith('toward_'):
        return 'adoption'

    # Reconceptualization: genuine shift in independent direction
    if shift is not None and shift > 0.15 and direction == 'independent':
        return 'reconceptualization'

    # Some shift but not clearly classified
    if shift is not None and shift > 0.1:
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
        except:
            result[0] = b''
    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)
    return result[0] if result[0] else b''


def _llm_call(system_prompt, user_prompt, config):
    import urllib.request, urllib.error, socket
    provider = config.get('provider', 'ollama')
    model = config.get('model', '')
    temp = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens', 300)
    old_timeout = socket.getdefaulttimeout()

    for attempt in range(2):
        try:
            if provider == 'ollama':
                socket.setdefaulttimeout(120)
                url = config.get('endpoint', 'http://localhost:11434') + '/api/chat'
                predict = max(max_tokens * 3, 1500) if 'qwen' in model.lower() else max_tokens
                body = json.dumps({
                    'model': model,
                    'messages': [{'role': 'system', 'content': system_prompt},
                                 {'role': 'user', 'content': user_prompt}],
                    'stream': False,
                    'options': {'temperature': temp, 'num_predict': predict}
                }).encode()
                req = urllib.request.Request(url, data=body,
                    headers={'Content-Type': 'application/json'})
                resp = urllib.request.urlopen(req, timeout=120)
                raw = _read_with_timeout(resp, timeout=120)
                text = json.loads(raw).get('message', {}).get('content', '')
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                socket.setdefaulttimeout(old_timeout)
                return text

            elif provider in ('openai', 'openrouter', 'custom'):
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
                    'messages': [{'role': 'system', 'content': system_prompt},
                                 {'role': 'user', 'content': user_prompt}],
                    'temperature': temp, 'max_tokens': max_tokens,
                }).encode()
                headers = {'Content-Type': 'application/json'}
                if key:
                    headers['Authorization'] = f'Bearer {key}'
                req = urllib.request.Request(url, data=body, headers=headers)
                resp = urllib.request.urlopen(req, timeout=60)
                raw = _read_with_timeout(resp, timeout=60)
                choices = json.loads(raw).get('choices', [])
                socket.setdefaulttimeout(old_timeout)
                return choices[0].get('message', {}).get('content', '') if choices else ''

            elif provider == 'anthropic':
                socket.setdefaulttimeout(60)
                key = os.environ.get('ANTHROPIC_API_KEY', '')
                body = json.dumps({
                    'model': model, 'max_tokens': max_tokens,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                    'temperature': temp,
                }).encode()
                req = urllib.request.Request('https://api.anthropic.com/v1/messages',
                    data=body, headers={
                        'Content-Type': 'application/json', 'x-api-key': key,
                        'anthropic-version': '2023-06-01',
                    })
                resp = urllib.request.urlopen(req, timeout=60)
                raw = _read_with_timeout(resp, timeout=60)
                blocks = json.loads(raw).get('content', [])
                socket.setdefaulttimeout(old_timeout)
                return ' '.join(b.get('text', '') for b in blocks if b.get('type') == 'text')

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
                parts = json.loads(raw).get('candidates', [{}])[0].get('content', {}).get('parts', [])
                socket.setdefaulttimeout(old_timeout)
                return parts[0].get('text', '') if parts else ''

        except Exception:
            socket.setdefaulttimeout(old_timeout)
            if attempt == 0:
                time.sleep(1)
            else:
                return ''
    socket.setdefaulttimeout(old_timeout)
    return ''


# ============================================================
# STRATEGIES — research-backed structural constraints
# ============================================================

STRATEGIES = {
    'default': {
        'name': 'Default',
        'system': 'Answer the question directly and honestly.',
    },
    # === Evidence-backed general strategies ===
    'devils_advocate': {
        'name': "Devil's Advocate",
        'system': 'Argue AGAINST the most likely answer. No hedging, no "both sides." Take the opposing position and defend it with your strongest arguments. Be specific and committed.',
        'prefix': 'The common answer is probably obvious. Argue against it:\n\n',
        'evidence': 'Lord, Lepper & Preston 1984 — "consider the opposite" eliminates anchoring',
    },
    'blind_spot': {
        'name': 'Blind Spot',
        'system': 'Identify exactly ONE thing most people overlook about this. Not a list. Name the blind spot, explain why people miss it, and what changes when you see it.',
        'prefix': 'What is the one thing most people miss:\n\n',
    },
    'first_principles': {
        'name': 'First Principles',
        'system': 'List the 2-3 assumptions everyone takes for granted. Question each one — what if it is wrong? Rebuild the answer from scratch without those assumptions.',
        'prefix': 'Break this down to first principles:\n\n',
        'evidence': 'Koriat 1980 — counterargument generation calibrates confidence',
    },
    'inversion': {
        'name': 'Inversion',
        'system': 'Answer the exact OPPOSITE question. If it asks how to succeed, explain how to guarantee failure. Be thorough. Let the contrast reveal what the direct answer misses.',
        'prefix': 'Answer the opposite:\n\n',
        'evidence': 'Mussweiler 2000 — considering the opposite eliminates anchoring in expert judgment',
    },
    'systems': {
        'name': 'Systems',
        'system': 'Ignore the direct answer. Focus on second-order and third-order effects only. What happens BECAUSE of the obvious answer? Follow the chain at least three steps.',
        'prefix': 'What are the downstream consequences:\n\n',
    },
    'stakeholder': {
        'name': 'Stakeholder',
        'system': 'Identify who is most HARMED by the conventional answer or approach. Explain entirely from their perspective. Give voice only to the one who loses.',
        'prefix': 'Who loses when we answer this the standard way:\n\n',
        'evidence': 'Galinsky & Moskowitz 2000 — perspective-taking reduces bias',
    },
    # === Research-specific strategies (evidence-backed) ===
    'pre_mortem': {
        'name': 'Pre-Mortem',
        'system': 'This research direction has been pursued and FAILED completely. The failure was predictable. Write the post-mortem: what went wrong, what warning signs were ignored, why the approach was flawed from the start. Be specific — name the exact failure mode.',
        'prefix': 'Assume this has already failed:\n\n',
        'evidence': 'Klein 2007 — prospective hindsight generates 30% more failure reasons, reduces overconfidence',
    },
    'alternative_hypothesis': {
        'name': 'Alt Hypothesis',
        'system': 'Name 3 genuinely different explanations for the same evidence. Not variations — structurally different causal mechanisms. For each, briefly state what data would distinguish it from the original.',
        'prefix': 'What else could explain this:\n\n',
        'evidence': 'Hirt & Markman 1995 — any alternative triggers debiasing simulation mindset',
    },
    'falsification': {
        'name': 'Falsification',
        'system': 'What specific, concrete, observable result would DISPROVE this? Name the exact experiment or observation that would force abandoning this position. If nothing could disprove it, explain why that is a serious problem.',
        'prefix': 'What would disprove this:\n\n',
        'evidence': 'Tetlock 2015 — superforecasters are 60% more accurate; falsification thinking is their key habit',
    },
    'adjacent_field': {
        'name': 'Adjacent Field',
        'system': 'Choose a completely different academic field. Describe how a researcher from that field would frame this same problem — different vocabulary, different mechanisms, different methods. Name the field and explain the reframing concretely.',
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
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s[0] for s in scored[:min(2, len(scored))]]
    remaining = [s[0] for s in scored if s[0] not in top]
    n_explore = min(n - len(top), len(remaining))
    explore_picks = random.sample(remaining, n_explore) if n_explore > 0 else []
    return (top + explore_picks)[:n]


def _generate_perspectives(question, strategies, config, quiet=False):
    provider = config.get('provider', 'ollama')
    results = {}
    _p = (lambda *a, **kw: None) if quiet else (lambda *a, **kw: print(*a, **kw))

    _p(f"  [1/{len(strategies)+1}] Default", end="", flush=True)
    default_resp = _llm_call(STRATEGIES['default']['system'], question,
        {**config, 'max_tokens': config.get('max_tokens', 300) + 100})
    results['default'] = default_resp
    _p(f" — done" if default_resp else " — failed")

    if provider in ('openai', 'anthropic', 'gemini', 'openrouter', 'custom'):
        lock = threading.Lock()
        def _gen(idx, key):
            s = STRATEGIES[key]
            resp = _llm_call(s['system'], s.get('prefix', '') + question, config)
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
        for i, key in enumerate(strategies):
            s = STRATEGIES[key]
            _p(f"  [{i+2}/{len(strategies)+1}] {s['name']}", end="", flush=True)
            resp = _llm_call(s['system'], s.get('prefix', '') + question, config)
            results[key] = resp
            _p(f" — {'done' if resp else 'failed'}")

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
        except:
            pass
    return None


def _load_state():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            if data.get('version') == VERSION:
                return data
        except:
            pass
    migrated = _migrate_legacy_state()
    return migrated if migrated else _new_state()


def _save_state(state):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    state['last_used'] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _log(msg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# ============================================================
# FEEDBACK
# ============================================================

def _update_weights(state, shown, rated, session_type, direction):
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
    for key in w:
        w[key] = w[key] * 0.99 + 1.0 * 0.01
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

    print(f"\n  PRISM")
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
    responses = _generate_perspectives(question, strategies, cfg)
    responses = {k: v for k, v in responses.items() if v}

    if 'default' not in responses:
        print("  Default response failed. Check your LLM configuration.")
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

    session = {
        'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:8],
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'human_before': human_before[:500] if human_before else None,
        'human_after': human_after[:500] if human_after else None,
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
        print(f"\n  Session logged. Run 'prism insights' for patterns.\n")
    _log(f"explore: q='{question[:50]}' type={session_type} conf={conf_before}→{conf_after}")


def check(conclusion):
    """Challenge an AI conclusion before committing."""
    cfg = _load_config()

    print(f"\n  PRISM — Challenge")
    print(f"  {'=' * 56}")
    _print_wrapped(conclusion, indent=2)
    print(f"  {'=' * 56}\n")

    print(f"  Generating challenges ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    provider = cfg.get('provider', 'ollama')
    results = {}

    if provider in ('openai', 'anthropic', 'gemini', 'openrouter', 'custom'):
        lock = threading.Lock()
        def _gen(key):
            s = STRATEGIES[key]
            resp = _llm_call(s['system'], s.get('prefix', '') + conclusion, cfg)
            with lock:
                results[key] = resp
        threads = [threading.Thread(target=_gen, args=(k,), daemon=True)
                   for k in CHECK_STRATEGIES]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
    else:
        for key in CHECK_STRATEGIES:
            s = STRATEGIES[key]
            print(f"  {s['name']}...", end="", flush=True)
            resp = _llm_call(s['system'], s.get('prefix', '') + conclusion, cfg)
            results[key] = resp
            print(f" done" if resp else " failed")

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
    print(f"\n  PRISM — Quick View\n  {question}\n")
    strategies = _select_strategies(state, cfg)
    print(f"  Generating ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    responses = _generate_perspectives(question, strategies, cfg)
    responses = {k: v for k, v in responses.items() if v}
    if 'default' not in responses:
        print("  Failed. Check LLM configuration.")
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
    print(f"\n  PRISM — Insights\n  {'─' * 40}\n  Sessions: {len(sessions)}")

    if len(sessions) < 3:
        print(f'\n  Need at least 3 sessions.\n  Run: prism "your question"\n')
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
            print("    Prism is creating productive doubt")
        elif avg > 1:
            print("    Caution: confidence rising after perspectives")

    # Strategy effectiveness by deep shift rate
    if len(sessions) >= 5:
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
    if len(conv) >= 10:
        recent = conv[-20:]
        nr = len(recent)
        xm = (nr - 1) / 2
        ym = sum(recent) / nr
        num = sum((i - xm) * (s - ym) for i, s in enumerate(recent))
        den = sum((i - xm) ** 2 for i in range(nr))
        slope = num / den if den > 0 else 0
        print(f"\n  Convergence (last {nr} sessions):")
        if slope < -0.01:
            print(f"    CONVERGING (slope: {slope:.4f}) — moving toward AI defaults")
        elif slope > 0.01:
            print(f"    DIVERGING (slope: {slope:.4f}) — increasingly independent")
        else:
            print(f"    STABLE (slope: {slope:.4f})")

    print()


def history():
    state = _load_state()
    sessions = state.get('sessions', [])
    print(f"\n  PRISM — History ({len(sessions)} sessions)\n  {'─' * 56}")
    if not sessions:
        print('  No sessions yet.\n')
        return
    for s in sessions[-10:]:
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
        print(f"\n  PRISM — Configuration\n  {'─' * 40}")
        for k, v in merged.items():
            src = '(project)' if k in project_cfg else '(global)' if k in global_cfg else '(auto)'
            print(f"  {k:>20}: {v}  {src}")
        print(f"\n  Set: prism config <key> <value>")
        print(f"  Global: {CONFIG_DIR / 'config.json'}")
        print(f"  Project: .prism.json\n")
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
        proj_file = Path.cwd() / '.prism.json'
        proj = {}
        if proj_file.exists():
            try:
                proj = json.loads(proj_file.read_text())
            except:
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

def get_perspectives(question, n=None):
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


def get_check(conclusion):
    """Challenge a conclusion. Returns structured JSON for integrations."""
    cfg = _load_config()
    provider = cfg.get('provider', 'ollama')
    results = {}

    if provider in ('openai', 'anthropic', 'gemini', 'openrouter', 'custom'):
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
    else:
        for key in CHECK_STRATEGIES:
            s = STRATEGIES[key]
            resp = _llm_call(s['system'], s.get('prefix', '') + conclusion, cfg)
            if resp:
                results[key] = resp

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

def setup(platform):
    """Set up integration with AI coding tools."""
    prism_path = os.path.realpath(__file__)

    if platform == 'install':
        _setup_install(prism_path)
    elif platform in ('claude', 'claude-code'):
        _setup_install(prism_path)
        _setup_claude_code()
    elif platform == 'codex':
        _setup_install(prism_path)
        _setup_codex()
    elif platform == 'cursor':
        _setup_install(prism_path)
        _setup_cursor()
    elif platform == 'copilot':
        _setup_install(prism_path)
        _setup_copilot()
    elif platform == 'windsurf':
        _setup_install(prism_path)
        _setup_windsurf()
    elif platform == 'all':
        _setup_install(prism_path)
        _setup_claude_code()
        _setup_codex()
        _setup_copilot()
        print(f"\n  All global integrations installed.\n")
    else:
        print(f"\n  PRISM — Setup")
        print(f"  {'─' * 40}")
        print(f"  prism setup install    # make 'prism' available globally")
        print(f"  prism setup claude     # Claude Code (/prism, /prism-check)")
        print(f"  prism setup codex      # Codex CLI")
        print(f"  prism setup cursor     # Cursor (in project dir)")
        print(f"  prism setup copilot    # GitHub Copilot")
        print(f"  prism setup windsurf   # Windsurf (in project dir)")
        print(f"  prism setup all        # install + claude + codex + copilot\n")


def _setup_install(prism_path):
    """Make 'prism' available as a system command."""
    bin_dir = Path.home() / '.local' / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / 'prism'

    # Make executable
    os.chmod(prism_path, os.stat(prism_path).st_mode | 0o755)

    # Create symlink
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(prism_path)

    # Check PATH
    on_path = str(bin_dir) in os.environ.get('PATH', '').split(':')

    print(f"\n  Installed: {link} -> {prism_path}")
    if on_path:
        print(f"  'prism' is on PATH. Ready to use.")
    else:
        print(f"\n  Add to ~/.bashrc or ~/.zshrc:")
        print(f'    export PATH="$PATH:{bin_dir}"')
        print(f"  Then: source ~/.bashrc")


def _setup_claude_code():
    """Create /prism and /prism-check slash commands for Claude Code."""
    cmd_dir = Path.home() / '.claude' / 'commands'
    cmd_dir.mkdir(parents=True, exist_ok=True)

    (cmd_dir / 'prism.md').write_text(
"""Generate divergent perspectives on this topic using Prism.

Run this command:
```bash
prism json "$ARGUMENTS"
```

Parse the JSON output. For each perspective show the strategy name, divergence score, and the key insight in 2-3 sentences. Then ask: "Do any of these change how you think about this?"
""")

    (cmd_dir / 'prism-check.md').write_text(
"""Challenge this conclusion using Prism before committing to it.

Run this command:
```bash
prism json --check "$ARGUMENTS"
```

Parse the JSON output. For each challenge (Pre-Mortem, Alt Hypothesis, Falsification, Blind Spot) summarize the key point in 1-2 sentences. Then assess: does the original conclusion still hold?
""")

    print(f"\n  Claude Code commands installed: {cmd_dir}")
    print(f"    /prism <question>          — divergent perspectives")
    print(f"    /prism-check <conclusion>  — challenge a conclusion")
    print(f"\n  NOTE: Restart Claude Code to pick up new commands.")
    print(f"  Alt: use ! prism check \"conclusion\" (always works, no restart)")


def _setup_codex():
    """Add Prism to Codex CLI instructions."""
    codex_dir = Path.home() / '.codex'
    codex_dir.mkdir(parents=True, exist_ok=True)
    f = codex_dir / 'instructions.md'

    block = """
## Prism

To challenge a conclusion: `prism json --check "the conclusion"`
To get perspectives: `prism json "the question"`
Run the command, parse JSON output, present each perspective concisely.
"""
    if f.exists():
        existing = f.read_text()
        if 'Prism' in existing:
            print(f"  Codex: Prism already configured in {f}")
            return
        f.write_text(existing.rstrip() + '\n' + block)
    else:
        f.write_text(block.strip() + '\n')
    print(f"  Codex: added Prism to {f}")


def _setup_cursor():
    """Create Cursor rule for Prism."""
    rules_dir = Path.cwd() / '.cursor' / 'rules'
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / 'prism.mdc').write_text(
"""---
description: Challenge conclusions and get diverse perspectives using Prism
globs:
alwaysApply: false
---

When asked to challenge a conclusion: `prism json --check "the conclusion"`
When asked for perspectives: `prism json "the question"`
Run the command, parse JSON, present each perspective in 2-3 sentences.
""")
    print(f"  Cursor: created {rules_dir / 'prism.mdc'}")


def _setup_copilot():
    """Add Prism to GitHub Copilot instructions."""
    gh_dir = Path.cwd() / '.github'
    gh_dir.mkdir(parents=True, exist_ok=True)
    f = gh_dir / 'copilot-instructions.md'

    block = """
## Prism

To challenge a conclusion: `prism json --check "the conclusion"`
To get perspectives: `prism json "the question"`
Run the command, parse JSON output, present each perspective concisely.
"""
    if f.exists():
        existing = f.read_text()
        if 'Prism' in existing:
            print(f"  Copilot: Prism already configured in {f}")
            return
        f.write_text(existing.rstrip() + '\n' + block)
    else:
        f.write_text(block.strip() + '\n')
    print(f"  Copilot: added Prism to {f}")


def _setup_windsurf():
    """Add Prism to Windsurf rules."""
    f = Path.cwd() / '.windsurfrules'
    block = """
## Prism

To challenge a conclusion: `prism json --check "conclusion"`
To get perspectives: `prism json "question"`
Parse JSON output and present concisely.
"""
    if f.exists():
        existing = f.read_text()
        if 'Prism' not in existing:
            f.write_text(existing.rstrip() + '\n' + block)
    else:
        f.write_text(block.strip() + '\n')
    print(f"  Windsurf: updated {f}")


# ============================================================
# MAIN
# ============================================================

def _main():
    args = sys.argv
    cmd = args[1] if len(args) > 1 else ''

    if cmd == 'explore':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            explore(q)
        else:
            print('  Usage: prism explore "question"')
    elif cmd == 'check':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            check(q)
        else:
            print('  Usage: prism check "AI conclusion to challenge"')
    elif cmd == 'quick':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            quick(q)
        else:
            print('  Usage: prism quick "question"')
    elif cmd == 'think':
        think()
    elif cmd == 'insights':
        insights()
    elif cmd == 'history':
        history()
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
    elif cmd in ('--help', '-h', 'help'):
        print(__doc__)
    elif cmd:
        explore(' '.join(args[1:]))
    else:
        think()


if __name__ == '__main__':
    _main()
