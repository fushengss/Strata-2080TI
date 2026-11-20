"""Deterministic synthetic prompts; no model/user/private material."""
import hashlib
import json

WORDS = ('the river carried old maps past quiet towns where people traded stories about storms bridges lanterns '
         'harvests letters clocks ships and distant mountains while children counted stars above the harbor').split()
TASK = ('Write only a very long English field diary describing many different harbor towns, their history, '
        'weather and daily work. Use numbered entries with substantial detail and continue for thousands of '
        'entries. Do not summarize, quote the source, or discuss your reasoning. Start the diary immediately.')


def prompt(words, fixture):
    assert isinstance(words, int) and words >= 0
    text = ' '.join(WORDS[(i * 7 + 8) % len(WORDS)] + ('.' if i % 17 == 16 else '') for i in range(words))
    return f'{fixture}: independent synthetic fixture.\n{text}\n{TASK}'


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def specs(fixture_id, cases=('primary', 'long')):
    result = []
    for case in cases:
        context = 10000 if case == 'primary' else 100000
        output = 4096 if case == 'primary' else 6144
        for lane in range(4):
            result.append({'name': f'{case}-resident-{lane}', 'case': case, 'role': 'resident', 'lane': lane,
                           'target_tokens': context, 'output_tokens': output,
                           'fixture': f'{fixture_id}-{case}-resident-{lane}'})
        for index, target in enumerate((10000, 60000) if case == 'primary' else (100000,)):
            result.append({'name': f'{case}-injected-{index}', 'case': case, 'role': 'injected', 'lane': None,
                           'target_tokens': target, 'output_tokens': 64,
                           'fixture': f'{fixture_id}-{case}-cold-{index}'})
    return result


def count_body(text, model):
    return {'model': model, 'messages': [{'role': 'user', 'content': text}],
            'thinking': {'type': 'disabled'}, 'max_tokens': 1}


def chat_body(text, model, output_tokens, seed=20261108, temperature=0.8):
    return {'model': model, 'messages': [{'role': 'user', 'content': text}], 'max_tokens': output_tokens,
            'temperature': temperature, 'top_p': 0.95, 'top_k': 20, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': False}}
