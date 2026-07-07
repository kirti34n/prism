#!/usr/bin/env python3
"""
Prism - a decision journal for the AI era.

Record what you thought before and after consulting an AI, get evidence-based
counterarguments to the AI's default answer, and track your own trajectory.
Research-backed structural constraints. Zero dependencies.

  prism "question"              # explore: position → perspectives → revised position
  prism check "AI conclusion"   # challenge an AI conclusion before committing
  prism research "question"     # deep analysis: 5 perspectives, 800 tokens, forced strategies
  prism quick "question"        # just show perspectives, no measurement
  prism think                   # random prompt → explore
  prism insights                # your thinking patterns over time
  prism revisit                 # look back at a past session - were you right?
  prism history                 # recent sessions
  prism config [key] [val]      # show or set configuration
  prism setup install           # how to install the 'prism' command
  prism setup claude            # Claude Code plugin (/prism, /prism-check)
  prism setup all               # all AI-tool integration paths
  prism json "question"         # machine-readable output
  prism json --check "concl"    # machine-readable check output
  prism reset                   # fresh start
  prism --version               # show version

Config:  .prism.json (project) → ~/.config/prism/config.json (global) → auto-detect
"""

from __future__ import annotations

import json, sys, time, hashlib, random, re, math, os, threading
from pathlib import Path
from datetime import datetime
from collections import Counter

VERSION = 3
__version__ = '3.0.0'

# ============================================================
# CONSTANTS
# ============================================================

# Conviction is self-reported on a 0-100 scale. A drop this large flags
# "destabilization" - ~2x the noise band of a 0-100 self-report, and larger
# than typical persuasion-RCT effects (5-15 pts), so it catches genuine doubt.
DESTABILIZATION_DROP = 20
INPUT_TRUNCATION_LIMIT = 500

# Operational limits
MAX_CONCURRENT_LLM = 8
MAX_SESSIONS = 500
OLLAMA_TIMEOUT = 120
API_TIMEOUT = 60
GEMINI_TIMEOUT = 90

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
                return json.loads(f.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError, ValueError):
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
            return json.loads(f.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return {}


def _save_global_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')


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

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _bow_distance(a: str, b: str) -> float:
    """Bag-of-words cosine distance. USED ONLY to order perspectives for display.
    Makes no claim about opinion change - see _classify_session for that."""
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


def _classify_session(self_category, conv_before, conv_after, moved_by):
    """Classify from self-report + conviction delta only. The user's own
    category is authoritative; text distance plays no part. First match wins."""
    if self_category == 'different_question':
        return 'reframing'
    if conv_before is not None and conv_after is not None:
        if conv_after - conv_before <= -DESTABILIZATION_DROP:
            return 'destabilization'
    if moved_by and self_category in ('shifted', 'switched'):
        return 'adoption'
    if self_category == 'switched':
        return 'switch'
    if self_category == 'shifted':
        return 'shift'
    if self_category == 'same':
        return 'unshaken'
    return 'unmeasured'


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
    if os.environ.get('PRISM_VERBOSE'):
        print(f"  [verbose] {msg}", flush=True)


_THINKING_MODELS = ('qwen3', 'deepseek-r1', 'qwq', 'magistral')


def _build_ollama(system_prompt, user_prompt, model, temp, max_tokens, config):
    url = config.get('endpoint', 'http://localhost:11434') + '/api/chat'
    body = {
        'model': model,
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': user_prompt}],
        'stream': False,
        'options': {'temperature': temp, 'num_predict': max_tokens},
    }
    # Thinking models otherwise spend the whole token budget reasoning inside
    # <think> and get cut off before the answer, leaving nothing after we strip
    # the block. Disabling thinking sends the full budget to the answer we use.
    if any(k in model.lower() for k in _THINKING_MODELS):
        body['think'] = False
    headers = {'Content-Type': 'application/json'}
    return url, json.dumps(body).encode(), headers, OLLAMA_TIMEOUT


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
# STRATEGIES - research-backed structural constraints
# ============================================================

_ACCURACY_GUARD = ('Only cite real, verifiable examples; if unsure whether an example '
                   'is real, say so rather than inventing one.')
_CALIBRATE = 'State this at the confidence the evidence supports.'

STRATEGIES = {
    'default': {
        'name': 'Default',
        'system': 'Answer the question directly. Give the most practical, well-reasoned answer you can. Include specific technologies, approaches, or steps - not vague principles. If there are trade-offs, name them concretely. ' + _ACCURACY_GUARD,
    },
    # === Evidence-backed general strategies ===
    'devils_advocate': {
        'name': "Devil's Advocate",
        'system': 'Argue against the answer you are shown. Identify its single strongest claim and refute it directly - explain the mechanism by which it fails, not just that it can fail. ' + _ACCURACY_GUARD + ' State objections at the confidence the evidence earns: label strong objections as strong and speculative ones as speculative. End by naming the one condition under which the conventional answer would still hold.',
        'prefix': 'The common answer is probably obvious. Argue against it:\n\n',
        'evidence': 'Lord, Lepper & Preston 1984 - "consider the opposite" eliminates anchoring',
    },
    'blind_spot': {
        'name': 'Blind Spot',
        'system': 'Identify exactly ONE hidden assumption or overlooked constraint that changes the entire framing - a structural blind spot that, once seen, makes the answer you were shown look naive. Show specifically how that answer depends on the assumption, and explain the mechanism that keeps people from seeing it. Give a concrete example of what goes wrong when it is missed. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'What is the one thing most people miss:\n\n',
    },
    'first_principles': {
        'name': 'First Principles',
        'system': 'Strip away every inherited assumption behind the answer you were shown. Name the 2-3 things "everyone knows" that might be wrong. For each, describe the specific scenario where it breaks down, cite a real precedent if you have one, and show what answer you get when you rebuild without that assumption. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'Break this down to first principles:\n\n',
        'evidence': 'Koriat 1980 - counterargument generation calibrates confidence',
    },
    'inversion': {
        'name': 'Inversion',
        'system': 'Answer the exact OPPOSITE question. If they ask how to succeed, write a detailed recipe for guaranteed failure - specific steps, specific mistakes, specific bad decisions. Then state explicitly what the answer you were shown was hiding that the inversion reveals. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'Answer the opposite:\n\n',
        'evidence': 'Mussweiler 2000 - considering the opposite eliminates anchoring in expert judgment',
    },
    'systems': {
        'name': 'Systems',
        'system': 'Set aside the direct answer and map its second- and third-order effects. What breaks, shifts, or emerges BECAUSE of the answer you were shown? Follow each causal chain at least three steps with specific examples, and name the feedback loops. Identify where the system will fight back against the intended change. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'What are the downstream consequences:\n\n',
    },
    'stakeholder': {
        'name': 'Stakeholder',
        'system': 'Identify who gets harmed, marginalized, or locked out by the answer you were shown. Tell the story from their perspective - their constraints, their frustrations, their alternatives - and name specific scenarios that make the friction concrete for someone who is not the assumed user. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'Who loses when we answer this the standard way:\n\n',
        'evidence': 'Galinsky & Moskowitz 2000 - perspective-taking reduces bias',
    },
    # === Research-specific strategies (evidence-backed) ===
    'pre_mortem': {
        'name': 'Pre-Mortem',
        'system': 'It is 18 months from now. The answer you were shown was followed and it FAILED. Write the post-mortem: name the specific failure mode, the early warning signs that were rationalized away, and the moment the team should have pivoted but did not. Be concrete - "the API rate limits hit at 10K users and there was no fallback," not "scalability was an issue." ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'Assume this has already failed:\n\n',
        'evidence': 'Klein 2007 - prospective hindsight generates 30% more failure reasons, reduces overconfidence',
    },
    'alternative_hypothesis': {
        'name': 'Alt Hypothesis',
        'system': 'Name 3 genuinely different explanations or approaches for the same problem - structurally different mechanisms, not variations on the answer you were shown. For each, state (a) the core insight that makes it work, (b) one specific scenario where it outperforms that answer, and (c) the exact test or metric that would distinguish between them. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'What else could explain this:\n\n',
        'evidence': 'Hirt & Markman 1995 - any alternative triggers debiasing simulation mindset',
    },
    'falsification': {
        'name': 'Falsification',
        'system': 'Design the exact test that would DISPROVE the answer you were shown. Name the specific metric, threshold, and scenario: "If X does not achieve Y under condition Z within timeframe W, this approach is wrong." If no test can disprove it, that is itself a red flag - explain why unfalsifiable plans are dangerous and what would make this one testable. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'What would disprove this:\n\n',
        'evidence': 'Tetlock 2015 - superforecasters are 60% more accurate; falsification thinking is their key habit',
    },
    'adjacent_field': {
        'name': 'Adjacent Field',
        'system': 'Pick a specific field that has already solved an analogous problem. Name the field, the specific technique or framework, and how it maps onto this problem in concrete terms, using their vocabulary. Describe what a practitioner from that field would do differently in the first week - at least one specific, actionable idea the answer you were shown would never generate. ' + _ACCURACY_GUARD + ' ' + _CALIBRATE,
        'prefix': 'How would a different field see this:\n\n',
        'evidence': 'Uzzi 2013 (Science, 17.9M papers) - atypical combinations produce 2x citation impact',
    },
}

STRATEGY_KEYS = [k for k in STRATEGIES if k != 'default']

# Fixed set for `check` command
CHECK_STRATEGIES = ['pre_mortem', 'alternative_hypothesis', 'falsification', 'blind_spot']


def _select_strategies(config):
    n = config.get('num_perspectives', 4)
    cfg = config.get('strategies', 'auto')
    if isinstance(cfg, list):
        valid = [k for k in cfg if k in STRATEGIES and k != 'default']
        if valid:
            return valid[:n]
    return random.sample(STRATEGY_KEYS, min(n, len(STRATEGY_KEYS)))


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
                contrastive = f"The conventional AI answer is:\n{default_resp[:400]}\n\nIdentify its key claim and refute it specifically.\n\n"
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
        'sessions': [],
    }


def _migrate_v2(data):
    """v2 → v3: keep every old session (tagged legacy so readers can skip it in
    metrics), drop the bandit weights. Pure and idempotent - no save inside."""
    for s in data.get('sessions', []):
        s.setdefault('schema', 'v2-legacy')
    data.pop('strategy_weights', None)
    data['version'] = VERSION
    return data


def _migrate_legacy_state():
    # Migration from local prism_state.json (very old format)
    legacy = Path(__file__).parent / 'prism_state.json'
    if legacy.exists() and not STATE_FILE.exists():
        try:
            old = json.loads(legacy.read_text(encoding='utf-8'))
            new = _new_state()
            new['sessions'] = old.get('sessions', [])
            for s in new['sessions']:
                s.setdefault('schema', 'v2-legacy')
            _save_state(new)
            return new
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    return None


def _load_state():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
            if data.get('version') == VERSION:
                return data
            # Any older/unrecognized-but-parseable version that still holds sessions
            # is migrated, not discarded - never silently drop a user's history.
            if data.get('sessions') is not None:
                data = _migrate_v2(data)
                _save_state(data)
                return data
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
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
    tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    tmp.replace(STATE_FILE)


def _log(msg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


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


def _read_conviction(prompt="  Conviction (0-100): "):
    if not sys.stdin.isatty():
        return None
    try:
        val = input(prompt).strip().rstrip('%').strip()
        if val.isascii() and val.isdigit():
            n = int(val)
            if 0 <= n <= 100:
                return n
    except (EOFError, KeyboardInterrupt):
        pass
    return None


def _read_self_category():
    if not sys.stdin.isatty():
        return None
    cats = {'1': 'same', '2': 'shifted', '3': 'switched', '4': 'different_question'}
    print("  How did your position change?")
    print("  (1=same, 2=shifted, 3=switched sides, 4=different question now, Enter=skip)")
    try:
        return cats.get(input("  > ").strip())
    except (EOFError, KeyboardInterrupt):
        return None


def _read_moved_by(shown_keys):
    if not sys.stdin.isatty() or not shown_keys:
        return None
    parts = ['0=default'] + [f"{i+1}={STRATEGIES.get(k, {}).get('name', k)}"
                             for i, k in enumerate(shown_keys)]
    print(f"  Which moved you most? ({', '.join(parts)}, Enter=none)")
    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice == '0':
        return 'default'
    if choice.isdigit() and 1 <= int(choice) <= len(shown_keys):
        return shown_keys[int(choice) - 1]
    return None


def _read_position(label):
    if not sys.stdin.isatty():
        return None
    print(label, flush=True)
    try:
        return input("  > ").strip() or None
    except (EOFError, KeyboardInterrupt):
        return None


def _strategies_for(cfg, is_research):
    strategies = _select_strategies(cfg)
    if is_research:
        must_have = ['falsification', 'adjacent_field', 'alternative_hypothesis']
        missing = [s for s in must_have if s not in strategies and s in STRATEGY_KEYS]
        replaceable = [s for s in strategies if s not in must_have]
        for s in missing:
            if replaceable:
                strategies[strategies.index(replaceable.pop())] = s
    return strategies


def _run_perspectives(question, cfg, is_research):
    """Generate, validate, rank for display. Returns (responses, divergences, shown_keys);
    responses is None on total failure, shown_keys is [] when only the default came back."""
    print(f"\n  Generating perspectives ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    strategies = _strategies_for(cfg, is_research)
    _verbose(f"strategies: {strategies}")
    responses = {k: v for k, v in _generate_perspectives(question, strategies, cfg).items() if v}
    if 'default' not in responses:
        print(f"  Default response failed ({cfg.get('provider')}/{cfg.get('model')}).")
        print("  Run 'prism config' to verify settings.")
        return None, None, None
    non_default = {k: v for k, v in responses.items() if k != 'default'}
    if not non_default:
        print(f"\n  {'─' * 56}\n  DEFAULT\n  {'─' * 56}")
        _print_wrapped(responses['default'], indent=4)
        return responses, {}, []
    divergences = {k: _bow_distance(responses['default'], v) for k, v in non_default.items()}
    shown_keys = sorted(divergences, key=divergences.get, reverse=True)[:cfg.get('num_shown', 3)]
    return responses, divergences, shown_keys


def _show_default_and_perspectives(responses, shown_keys):
    print(f"\n  {'─' * 56}\n  DEFAULT ANSWER\n  {'─' * 56}")
    _print_wrapped(responses['default'], indent=4)
    for i, key in enumerate(shown_keys):
        name = STRATEGIES.get(key, {}).get('name', key)
        print(f"\n  {'─' * 56}\n  {i+1}. {name.upper()}\n  {'─' * 56}")
        _print_wrapped(responses[key], indent=4)


def _rebuttal_reply(question, key, perspective, pushback, cfg):
    user = (f"Question: {question}\n\nYour earlier position:\n{perspective[:400]}\n\n"
            f"The user pushes back:\n{pushback}\n\n"
            "Respond once, under 150 words. Concede what is right in the pushback; "
            "defend only what survives it.")
    return _llm_call(STRATEGIES[key]['system'], user, cfg)


def _rebuttal_round(question, shown_keys, responses, cfg):
    """One optional follow-up: user rebuts a perspective, that strategy replies once."""
    if not sys.stdin.isatty() or not shown_keys:
        return None
    print(f"\n  {'─' * 56}")
    print(f"  Push back on a perspective? (1-{len(shown_keys)} to respond, Enter to skip)")
    try:
        choice = input("  > ").strip()
        if not (choice.isdigit() and 1 <= int(choice) <= len(shown_keys)):
            return None
        print("  Your pushback:", flush=True)
        pushback = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not pushback:
        return None
    key = shown_keys[int(choice) - 1]
    reply = _rebuttal_reply(question, key, responses[key], pushback, cfg)
    name = STRATEGIES.get(key, {}).get('name', key)
    print(f"\n  {'─' * 56}\n  {name.upper()} RESPONDS\n  {'─' * 56}")
    _print_wrapped(reply or "(no response)", indent=4)
    return {'strategy': key, 'text': pushback[:INPUT_TRUNCATION_LIMIT]}


def _build_session(question, pos_before, conv_before, pos_after, conv_after,
                   self_category, moved_by, session_type, shown_keys, divergences, rebuttal):
    trunc = INPUT_TRUNCATION_LIMIT
    wording = _bow_distance(pos_before, pos_after) if pos_before and pos_after else None
    return {
        'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:8],
        'schema': 'v3',
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'position_before': pos_before[:trunc] if pos_before else None,
        'position_after': pos_after[:trunc] if pos_after else None,
        'conviction_before': conv_before,
        'conviction_after': conv_after,
        'self_category': self_category,
        'moved_by': moved_by,
        'session_type': session_type,
        'strategies_shown': shown_keys,
        'divergences': {k: round(v, 4) for k, v in divergences.items()},
        'rebuttal': rebuttal,
        'wording_change': round(wording, 4) if wording is not None else None,
    }


def _print_measurement(session):
    st = session['session_type']
    desc = {
        'reframing': 'You changed the question - frame shift',
        'destabilization': 'Conviction dropped sharply - productive doubt',
        'adoption': 'Moved toward a model answer - check if genuine',
        'switch': 'You flipped your stance',
        'shift': 'You moved, same side',
        'unshaken': 'No significant change',
    }
    print(f"\n  {'=' * 56}\n  MEASUREMENT\n  {'=' * 56}")
    print(f"\n  Session: {st}")
    if desc.get(st):
        print(f"  {desc[st]}")
    cb, ca = session['conviction_before'], session['conviction_after']
    if cb is not None and ca is not None:
        print(f"  Conviction: {cb} → {ca} ({ca - cb:+d})")
    if session.get('moved_by'):
        mb = session['moved_by']
        name = 'Default answer' if mb == 'default' else STRATEGIES.get(mb, {}).get('name', mb)
        print(f"  Moved by: {name}")


def explore(question):
    """Position + conviction → perspectives → rebuttal → revised position → classify."""
    state = _load_state()
    cfg = _load_config()
    print(f"\n  PRISM\n  {'=' * 56}\n  {question}\n  {'=' * 56}\n")

    position_before = _read_position("  Your position (before seeing anything):")
    conviction_before = _read_conviction() if position_before else None

    is_research = os.environ.pop('PRISM_RESEARCH', None)
    if is_research:
        cfg = {**cfg, 'max_tokens': 800, 'num_perspectives': 5, 'num_shown': 5}
    responses, divergences, shown_keys = _run_perspectives(question, cfg, is_research)
    if responses is None or not shown_keys:
        return

    _show_default_and_perspectives(responses, shown_keys)
    rebuttal = _rebuttal_round(question, shown_keys, responses, cfg)

    position_after = _read_position(f"\n  {'─' * 56}\n  Your position now:")
    conviction_after = _read_conviction("  Conviction now (0-100): ") if position_after else None
    self_category = _read_self_category() if position_after else None
    moved_by = _read_moved_by(shown_keys) if position_after else None

    session_type = _classify_session(self_category, conviction_before, conviction_after, moved_by)
    session = _build_session(question, position_before, conviction_before, position_after,
                             conviction_after, self_category, moved_by, session_type,
                             shown_keys, divergences, rebuttal)
    if session_type != 'unmeasured':
        _print_measurement(session)
    state.setdefault('sessions', []).append(session)
    _save_state(state)
    n = len(state['sessions'])
    if n < 5:
        print(f"\n  Session logged ({n}/5 for first insights).\n")
    else:
        print(f"\n  Session logged. Run 'prism insights' for patterns.\n")
    _log(f"explore: q='{question[:50]}' type={session_type} conv={conviction_before}→{conviction_after}")


def check(conclusion):
    """Challenge an AI conclusion before committing."""
    cfg = _load_config()

    print(f"\n  PRISM - Challenge")
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

    # Log check session to state
    state = _load_state()
    session = {
        'id': hashlib.sha256(str(time.time()).encode()).hexdigest()[:8],
        'schema': 'v3',
        'timestamp': datetime.now().isoformat(),
        'question': conclusion[:INPUT_TRUNCATION_LIMIT],
        'session_type': 'check',
        'strategies_shown': [k for k in CHECK_STRATEGIES if k in results and results[k]],
    }
    state.setdefault('sessions', []).append(session)
    _save_state(state)
    _log(f"check: '{conclusion[:50]}'")


def quick(question):
    cfg = _load_config()
    print(f"\n  PRISM - Quick View\n  {question}\n")
    strategies = _select_strategies(cfg)
    print(f"  Generating ({cfg.get('provider')}/{cfg.get('model')})...", flush=True)
    responses = _generate_perspectives(question, strategies, cfg)
    responses = {k: v for k, v in responses.items() if v}
    if 'default' not in responses:
        print(f"  Failed ({cfg.get('provider')}/{cfg.get('model')}). Run 'prism config' to verify.")
        return
    divergences = {k: _bow_distance(responses['default'], v)
                   for k, v in responses.items() if k != 'default'}
    sorted_keys = sorted(divergences, key=divergences.get, reverse=True)
    print(f"\n  {'─' * 56}\n  DEFAULT\n  {'─' * 56}")
    _print_wrapped(responses['default'], indent=4)
    for key in sorted_keys[:cfg.get('num_shown', 3)]:
        s = STRATEGIES.get(key, {})
        print(f"\n  {'─' * 56}\n  {s.get('name', key).upper()}\n  {'─' * 56}")
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


def _insight_categories(sessions):
    desc = {'reframing': 'changed the question', 'destabilization': 'conviction shaken',
            'adoption': 'moved toward AI', 'switch': 'flipped stance',
            'shift': 'moved, same stance', 'unshaken': 'no change', 'unmeasured': 'not measured'}
    types = Counter(s.get('session_type') for s in sessions if s.get('session_type'))
    if types:
        print(f"\n  Session types:")
        for t, count in types.most_common():
            print(f"    {t:>16}: {count}  ({desc.get(t, t)})")


def _insight_conviction(sessions):
    pairs = [(s['conviction_before'], s['conviction_after']) for s in sessions
             if s.get('conviction_before') is not None and s.get('conviction_after') is not None]
    if not pairs:
        return
    avg = sum(a - b for b, a in pairs) / len(pairs)
    print(f"\n  Conviction change: {avg:+.1f} average ({len(pairs)} measured)")
    if avg < -5:
        print("    Prism is creating productive doubt")
    elif avg > 5:
        print("    Caution: conviction rising after perspectives")


def _insight_adoption(sessions):
    classified = [s for s in sessions if s.get('session_type') not in (None, 'unmeasured')]
    recent = classified[-10:]
    if not recent:
        return
    adopted = sum(1 for s in recent
                  if s.get('moved_by') and s.get('self_category') in ('shifted', 'switched'))
    rate = adopted / len(recent)
    print(f"\n  Recent adoption: {rate:.0%} of last {len(recent)} classified sessions")
    if rate > 0.5:
        print("    Most recent changes moved toward AI positions - bring outside sources")


def _insight_moved(sessions):
    moved = Counter(s['moved_by'] for s in sessions if s.get('moved_by'))
    if not moved:
        return
    print(f"\n  What moved you:")
    for key, count in moved.most_common(6):
        name = 'Default answer' if key == 'default' else STRATEGIES.get(key, {}).get('name', key)
        print(f"    {name:>16}: {count}")


def _insight_revisits(sessions):
    outcomes = [s['revisit']['outcome'] for s in sessions if s.get('revisit')]
    if len(outcomes) < 3:
        return
    c = Counter(outcomes)
    decided = c['right'] + c['wrong']
    rate = f", {c['right']/decided:.0%} right of decided" if decided else ""
    print(f"\n  Revisits: {len(outcomes)} - {c['right']} right, {c['wrong']} wrong, "
          f"{c['unclear']} unclear{rate}")


def insights():
    state = _load_state()
    sessions = state.get('sessions', [])
    v3 = [s for s in sessions if s.get('schema') == 'v3']
    explore3 = [s for s in v3 if s.get('session_type') != 'check']
    n_legacy = sum(1 for s in sessions if s.get('schema') == 'v2-legacy')
    print(f"\n  PRISM - Insights\n  {'─' * 40}\n  Sessions: {len(sessions)}")
    if n_legacy:
        print(f"  ({n_legacy} legacy v2 sessions excluded from metrics)")
    if len(explore3) < 3:
        print(f'\n  Need at least 3 explore sessions.\n  Run: prism "your question"\n')
        return
    _insight_categories(explore3)
    _insight_conviction(explore3)
    _insight_adoption(explore3)
    _insight_moved(explore3)
    _insight_revisits(v3)
    print()


def history(count=10):
    state = _load_state()
    sessions = state.get('sessions', [])
    print(f"\n  PRISM - History ({len(sessions)} sessions)\n  {'─' * 56}")
    if not sessions:
        print('  No sessions yet.\n')
        return
    for s in sessions[-count:]:
        cb = s.get('conviction_before', s.get('confidence_before'))
        ca = s.get('conviction_after', s.get('confidence_after'))
        conv = f"{cb}→{ca}" if cb is not None and ca is not None else "-"
        stype = (s.get('session_type') or '')[:14]
        print(f"  [{conv:>7}] {stype:>14} | {s['question'][:35]}")
    print()


def _revisit_candidate(sessions):
    for s in sessions:
        if s.get('session_type') == 'check' or 'revisit' in s:
            continue
        if s.get('position_after') or s.get('human_after'):
            return s
    return None


def revisit():
    state = _load_state()
    s = _revisit_candidate(state.get('sessions', []))
    if not s:
        print("\n  Nothing to revisit yet.\n")
        return
    cb = s.get('conviction_before', s.get('confidence_before'))
    ca = s.get('conviction_after', s.get('confidence_after'))
    print(f"\n  PRISM - Revisit\n  {'─' * 56}")
    print(f"  {(s.get('timestamp') or '')[:10]} | {s['question']}")
    print(f"  Before: {s.get('position_before') or s.get('human_before') or '(none)'}  (conviction {cb})")
    print(f"  After:  {s.get('position_after') or s.get('human_after') or '(none)'}  (conviction {ca})")
    if not sys.stdin.isatty():
        return
    try:
        print("\n  Looking back: was your revised position right? (y/n/unclear)")
        ans = input("  > ").strip().lower()
        note = input("  Note (Enter to skip):\n  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    outcome = {'y': 'right', 'yes': 'right', 'n': 'wrong', 'no': 'wrong',
               'u': 'unclear', 'unclear': 'unclear'}.get(ans)
    if not outcome:
        print("  Skipped.\n")
        return
    s['revisit'] = {'timestamp': datetime.now().isoformat(),
                    'outcome': outcome, 'note': note[:INPUT_TRUNCATION_LIMIT] or None}
    _save_state(state)
    print(f"  Logged: {outcome}.\n")


def config_cmd(args):
    global_cfg = _load_global_config()
    project_cfg = _find_project_config()
    merged = _load_config()
    if not args:
        print(f"\n  PRISM - Configuration\n  {'─' * 40}")
        for k, v in merged.items():
            src = '(project)' if k in project_cfg else '(global)' if k in global_cfg else '(auto)'
            print(f"  {k:>20}: {v}  {src}")
        print(f"\n  Set: prism config <key> <value>")
        print(f"  Global: {CONFIG_DIR / 'config.json'}")
        print(f"  Project: .prism.json\n")
        mode = merged.get('strategies', 'auto')
        print(f"  Strategies ({mode} mode):")
        for k in STRATEGY_KEYS:
            ev = STRATEGIES[k].get('evidence', '')
            ev_str = f"  [{ev[:40]}]" if ev else ""
            print(f"    {k:>22} ({STRATEGIES[k]['name']}){ev_str}")
        print()
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
        proj_file = Path.cwd() / '.prism.json'
        proj = {}
        if proj_file.exists():
            try:
                proj = json.loads(proj_file.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        proj[key] = val
        proj_file.write_text(json.dumps(proj, indent=2), encoding='utf-8')
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
    cfg = _load_config()
    strategies = _select_strategies(cfg)
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
                'divergence': round(_bow_distance(responses['default'], responses[key]), 4),
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

# Prism does not write into other tools' internal config. Integration lives as
# committed files: the Claude Code plugin marketplace (.claude-plugin/) and the
# Agent Skills SKILL.md standard (skills/). Static files survive tool updates;
# mimicking a tool's undocumented cache format does not. See README.

_MARKETPLACE_REPO = 'kirti34n/prism'

_CLAUDE_STEPS = f"""
  Claude Code - add Prism as a plugin (survives updates, no file injection):

    /plugin marketplace add {_MARKETPLACE_REPO}
    /plugin install prism@prism

  Gives you /prism, /prism-check, and the auto-suggest skill.
"""

_SKILLS_STEPS = """
  Cursor, Copilot, Gemini CLI, Codex and other Agent-Skills tools:

    Point your tool at this repo's skills/ folder (the SKILL.md standard),
    or copy skills/prism into your tool's skills directory.
    See the README "AI-tool integration" section for per-tool paths.
"""


def setup(platform):
    """Print the standards-based integration steps (Prism no longer injects
    files into other tools - see README for why)."""
    if platform == 'install':
        _setup_install()
    elif platform in ('claude', 'claude-code'):
        print(_CLAUDE_STEPS)
    elif platform == 'all':
        print(_CLAUDE_STEPS)
        print(_SKILLS_STEPS)
    elif platform in ('cursor', 'codex', 'copilot', 'windsurf', 'kiro', 'gemini', 'augment'):
        print(_SKILLS_STEPS)
    else:
        _setup_help()


def _setup_help():
    print(f"\n  PRISM - Setup\n  {'─' * 40}")
    print("  prism setup install    # how to install the 'prism' command")
    print("  prism setup claude     # Claude Code plugin (/prism, /prism-check)")
    print("  prism setup all        # every AI-tool integration path\n")


def _setup_install():
    """Tell the user the right install command for their machine. No symlinks:
    the PyPI console-script entry point works cross-platform, Windows included."""
    import shutil
    if shutil.which('prism'):
        print("\n  'prism' is already on PATH. You're set.\n")
        return
    print("\n  Install Prism as a global command:\n")
    if shutil.which('pipx'):
        print("    pipx install prism-think")
    elif shutil.which('uv'):
        print("    uv tool install prism-think")
    else:
        print("    pipx install prism-think          # recommended (pip install pipx first)")
        print("    pip install --user prism-think    # fallback")
    print()


# ============================================================
# MAIN
# ============================================================

def _force_utf8_output():
    # The box-drawing dividers crash on a legacy Windows console (cp1252). Switch
    # stdout/stderr to UTF-8 so the CLI renders everywhere; harmless if already UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass


def _main():
    _force_utf8_output()
    args = sys.argv
    # Extract --verbose/-v flag
    verbose = '--verbose' in args or '-v' in args
    args = [a for a in args if a not in ('--verbose', '-v')]
    if verbose:
        os.environ['PRISM_VERBOSE'] = '1'

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
    elif cmd == 'research':
        q = ' '.join(args[2:]) if len(args) > 2 else None
        if q:
            os.environ['PRISM_RESEARCH'] = '1'
            explore(q)
        else:
            print('  Usage: prism research "question"')
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
    elif cmd == 'revisit':
        revisit()
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
        print(f"prism {__version__}")
    elif cmd in ('--help', '-h', 'help'):
        print(__doc__)
    elif cmd:
        explore(' '.join(args[1:]))
    else:
        think()


if __name__ == '__main__':
    try:
        _main()
    except KeyboardInterrupt:
        sys.exit(130)
