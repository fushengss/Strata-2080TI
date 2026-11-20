#!/usr/bin/env python3
"""Existing numeric SSE client plus in-memory synthetic assistant continuation.

No assistant text is persisted. A captured reply is checked against the existing
visible-channel hash before it is used as the next conversation's assistant turn.
Resident requests use the identical body, parsing and timeout semantics as v1.
"""
import hashlib
import http.client
import json
import time
import stream_client as base
from stream_client import CHANNELS, extract_delta, summarize_request, save

CAPTURE_LIMIT_BYTES=1<<20

def append_capture(buffer,part):
    if len(buffer)+len(part)>CAPTURE_LIMIT_BYTES:
        raise RuntimeError('synthetic_capture_limit')
    buffer.extend(part)

class Client(base.Client):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.captured_replies={}

    def request(self, name, role, lane, nominal, output_tokens, fixture, planned_epoch=None, messages=None, capture_response=False):
        prompt = base.synthetic_prompt(nominal, fixture)
        body = {'model': self.model, 'messages': messages if messages is not None else [{'role': 'user', 'content': prompt}],
                'stream': True, 'stream_options': {'include_usage': True}, 'max_tokens': output_tokens,
                'temperature': self.plan['temperature'], 'seed': self.plan['seed'],
                'chat_template_kwargs': {'enable_thinking': self.plan.get('thinking_enabled', True)}}
        for key in ('top_p', 'top_k'):
            if key in self.plan:
                body[key] = self.plan[key]
        raw = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        start_epoch, start = time.time(), time.monotonic()
        record = {'name': name, 'role': role, 'lane': lane, 'nominal_prompt_tokens': nominal,
                  'prompt_tolerance': self.plan['prompt_count_tolerance_fraction'], 'max_tokens': output_tokens,
                  'sampling_profile': self.plan['sampling_profile'],
                  'request_sha256': hashlib.sha256(raw).hexdigest(), 'start_epoch': start_epoch,
                  'planned_epoch': planned_epoch, 'schedule_delay_s': max(0, start_epoch - planned_epoch) if planned_epoch else None,
                  'chunks': [], 'usage': {}, 'saw_done': False, 'error': None, 'finish_reason': None}
        state = {'role': role, 'lane': lane, 'start': start, 'first': None, 'last': None, 'socket': None}
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        output_hashes = {channel: hashlib.sha256() for channel in CHANNELS if channel != 'output'}
        captured = bytearray()
        with self.lock:
            self.active[name] = state
        try:
            connection.connect()
            with self.lock:
                state['socket'] = connection.sock
            connection.sock.settimeout(self.plan['request_total_timeout_seconds'])
            connection.request('POST', '/v1/chat/completions', body=raw, headers=self.headers)
            response = connection.getresponse()
            record['http_status'] = response.status
            if response.status != 200:
                raise RuntimeError('HTTP_error')  # Never save the response body or exception message.
            with (self.out / (name + '.chunks.jsonl')).open('w') as journal:
                pending = []
                for raw_line in response:
                    line = raw_line.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'):
                        pending.append(line[5:].lstrip())
                        continue
                    if line or not pending:
                        continue
                    received = time.monotonic()
                    payload, pending = '\n'.join(pending), []
                    if payload == '[DONE]':
                        record['saw_done'] = True
                        break
                    data = json.loads(payload)
                    if data.get('error'):
                        raise RuntimeError('API_error')
                    if isinstance(data.get('timings'), dict):
                        record['native_timings'] = {k: v for k, v in data['timings'].items()
                            if k in ('cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second',
                                     'predicted_n', 'predicted_ms', 'predicted_per_token_ms', 'predicted_per_second',
                                     'draft_n', 'draft_n_accepted') and (v is None or type(v) in (int, float))}
                    if isinstance(data.get('usage'), dict):
                        usage = data['usage']
                        record['usage'] = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens') if isinstance(usage.get(key), int)}
                        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                        if isinstance(cached, int):
                            record['usage']['prompt_tokens_details'] = {'cached_tokens': cached}
                    for choice in data.get('choices') or []:
                        if choice.get('finish_reason'):
                            reason = choice['finish_reason']
                            record['finish_reason'] = reason if reason in ('stop', 'length', 'tool_calls', 'function_call', 'cancel', 'content_filter') else 'unknown'
                    channels, counts, parts = extract_delta(data)
                    if not channels:
                        continue
                    now = received
                    event = {'elapsed_s': now - start, 'epoch': start_epoch + now - start,
                             'channels': channels, 'bytes': counts,
                             'previous_output_gap_s': now - state['last'] if state['last'] is not None else None}
                    record['chunks'].append(event)
                    for channel, value in parts.items():
                        output_hashes[channel].update(value)
                    if capture_response:
                        append_capture(captured,parts['visible_content'])
                    with self.lock:
                        state['first'] = state['first'] if state['first'] is not None else now
                        state['last'] = now
                    journal.write(json.dumps(event, separators=(',', ':')) + '\n')
        except Exception as error:
            record['error'] = {'kind': type(error).__name__}  # No server text, credentials, or generated content.
        finally:
            connection.close()
            record['elapsed_s'] = time.monotonic() - start
            record['end_epoch'] = start_epoch + record['elapsed_s']
            record['channel_output_sha256'] = {channel: digest.hexdigest() for channel, digest in output_hashes.items()}
            record['output_sha256'] = hashlib.sha256(json.dumps(record['channel_output_sha256'], sort_keys=True).encode()).hexdigest()
            if capture_response:
                self.captured_replies[name] = captured.decode('utf-8')
                record['captured_visible_sha256'] = hashlib.sha256(captured).hexdigest()
                record['captured_visible_bytes'] = len(captured)
            record['hash_note'] = 'Per-channel text hashes do not depend on SSE chunk boundaries; combined hash covers these three digests.'
            if state.get('cancel_reason'):
                record['error'] = {'kind': state['cancel_reason']}
            elif not record['saw_done'] and record['error'] is None:
                record['error'] = {'kind': 'incomplete_stream'}
            summarize_request(record)
            save(self.out / (name + '.json'), record)
            with self.lock:
                self.records.append(record)
                self.active.pop(name, None)
        return record

