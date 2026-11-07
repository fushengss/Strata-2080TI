"""Opt-in native thinking cap: CPU fakes, including real BT/BDONE transport; no GPU/HTTP."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

from serve.server import (ByteTokenizer, EngineDied, EngineNonFinite, MockEngine, REASONING_WRAP_UP,
                          Service, StrataEngine, clean_shared_defaults)


class Tokenizer(ByteTokenizer):
    def token_bytes(self, token):
        return self.SPECIALS[token - 256].encode() if token >= 256 else bytes([token])


class NativeFake(StrataEngine):
    """Resume by absolute prompt position; record completion only after the final yield is consumed."""
    def __init__(self, tok, text):
        self.tok, self.base = tok, tok.encode("P:")
        self.script = tok.encode(text) + tok.encode("<|im_end|>", parse_special=True)
        self.answer = tok.encode("Answer.") + tok.encode("<|im_end|>", parse_special=True)
        self.wrap = tok.encode(REASONING_WRAP_UP, parse_special=True)
        self.max_context, self.info, self.last = 65536, {}, {}
        self.split_engine = False
        self.calls, self.closed_early, self.completed = [], [], []
        self.fail = None
        self.cancel_at = None

    def alive(self):
        return True

    def death_note(self):
        return ""

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        index = len(self.calls)
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        tail = ids[len(self.base):]
        wrapped = next((i for i in range(len(tail)) if tail[i:i + len(self.wrap)] == self.wrap), None)
        remaining = self.script[len(tail):] if wrapped is None else self.answer[len(tail) - wrapped - len(self.wrap):]
        self.last = {"generated": 1, "prompt_ms": 1.0, "decode_ms": 0.0,
                     "prompt_tokens": len(ids), "reused": 0, "prompt_read": len(ids)}
        emitted, ended = 0, False
        try:
            for t in remaining[:max_new]:
                if cancel.is_set():
                    return
                emitted += 1
                if self.cancel_at == (index, emitted):
                    cancel.set()
                yield t
            if self.fail is not None:
                raise self.fail
            ended = True
            self.completed.append(index)
            self.last = {**self.last, "generated": emitted, "decode_ms": 2.0 * emitted,
                         "finish": "length" if emitted == max_new else "stop"}
        finally:
            if emitted and remaining[emitted - 1] in self.tok.encode("<|im_end|>", parse_special=True):
                # Native transport drains the terminal DONE/BDONE when its EOS is consumed.
                self.last = {**self.last, "generated": emitted, "decode_ms": 2.0 * emitted, "finish": "stop"}
            if not ended:
                self.closed_early.append(index)


class BudgetCap(unittest.TestCase):
    def run_case(self, text="x" * 300, *, cap=True, budget=20, maximum=400, sampling=None,
                 configure=None, thinking=True, force=None):
        tok = Tokenizer()
        engine = NativeFake(tok, text)
        service = Service(engine, tok, None)
        service.answer_room = False
        service.reasoning_budget_engine_cap = cap
        service.reasoning_budget_tokens = budget
        if configure:
            configure(service, engine)
        request = {"seed": 17, "temperature": 0.8, **(sampling or {})}
        cancel = threading.Event()
        with contextlib.redirect_stdout(io.StringIO()):
            events = list(service.run(engine.base, thinking, [], maximum, request, cancel, force=force))
        text_by_kind = {kind: "".join(e.text or "" for typ, e in events if typ == "event" and e.kind == kind)
                        for kind in ("reasoning", "content")}
        return service, engine, events[-1][1], text_by_kind

    def test_default_off_and_opt_in_wrap_have_same_text_usage(self):
        off = self.run_case(cap=False)
        on = self.run_case()
        self.assertFalse(Service(MockEngine(Tokenizer(), "x"), Tokenizer(), None).reasoning_budget_engine_cap)
        self.assertEqual(off[3], on[3])
        self.assertEqual(off[2]["completion_tokens"], on[2]["completion_tokens"])
        self.assertEqual(off[1].calls[0][1], 400)
        self.assertEqual(on[1].calls[0][1], 20)
        self.assertIn(0, on[1].completed)  # native length-ending statistics consumed before wrap
        self.assertNotIn(0, on[1].closed_early)
        self.assertEqual(on[0].history[-1]["engine_generated"], 20 + len(on[1].answer))

    def test_natural_think_end_keeps_full_answer_after_artificial_cap(self):
        text = "abc</think>\n\n" + "answer " * 20
        off = self.run_case(text, cap=False, budget=30)
        on = self.run_case(text, budget=30)
        self.assertEqual(off[3], on[3])
        self.assertEqual(off[2]["completion_tokens"], on[2]["completion_tokens"])
        self.assertEqual(on[2]["finish"], "stop")
        self.assertEqual(on[1].calls[1][0], on[1].base + on[1].script[:30])
        self.assertEqual(on[1].calls[1][2]["seed"], 17)

    def test_utf8_boundary_advances_one_token_until_clean(self):
        result = self.run_case("é" + "x" * 100, budget=1)
        self.assertEqual([c[1] for c in result[1].calls[:2]], [1, 1])
        self.assertTrue(result[3]["reasoning"].startswith("é"))
        self.assertNotIn("�", result[3]["reasoning"])
        self.assertEqual(result[3]["content"], "Answer.")

    def test_partial_closing_tag_keeps_parser_and_continues_answer(self):
        text = "abc</think>\n\nAn intact answer."
        result = self.run_case(text, budget=4)
        self.assertEqual(result[3], {"reasoning": "abc", "content": "An intact answer."})
        self.assertTrue(all(c[1] == 1 for c in result[1].calls[1:-1]))
        self.assertEqual(result[2]["finish"], "stop")

    def test_real_eos_at_cap_is_not_continued(self):
        result = self.run_case("abc", budget=4)
        self.assertEqual(len(result[1].calls), 1)
        self.assertEqual(result[2]["finish"], "stop")

    def test_stop_string_is_not_continued(self):
        result = self.run_case("abc</think>\n\nSTOPmore", budget=30, sampling={"stop": ["STOP"]})
        self.assertEqual(len(result[1].calls), 1)
        self.assertEqual(result[2]["finish"], "stop")

    def test_cancellation_at_cap_is_not_continued(self):
        result = self.run_case(configure=lambda s, e: setattr(e, "cancel_at", (0, 20)))
        self.assertEqual(len(result[1].calls), 1)
        self.assertEqual(result[2]["finish"], "cancel")

    def test_nonfinite_at_cap_drain_does_not_wrap(self):
        result = self.run_case(configure=lambda s, e: setattr(e, "fail", EngineNonFinite("fake nonfinite")))
        self.assertEqual(len(result[1].calls), 1)
        self.assertEqual(result[3]["content"], "")

    def test_engine_death_during_cap_drain_propagates(self):
        with self.assertRaises(EngineDied):
            self.run_case(configure=lambda s, e: setattr(e, "fail", EngineDied("fake death")))

    def test_penalties_fall_back_without_mutating_sampling(self):
        for penalty in ({"presence_penalty": 0.1}, {"frequency_penalty": 0.1}, {"repetition_penalty": 1.1}):
            with self.subTest(penalty=penalty):
                result = self.run_case(sampling=penalty)
                self.assertEqual(result[1].calls[0][1], 400)
                self.assertEqual(result[1].calls[0][2], result[1].calls[1][2])
        neutral = self.run_case(sampling={"presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1})
        self.assertEqual(neutral[1].calls[0][1], 20)

    def test_no_budget_no_thinking_and_other_backends_retain_original_path(self):
        self.assertEqual(self.run_case(budget=0)[1].calls[0][1], 400)
        self.assertEqual(self.run_case(thinking=False)[1].calls[0][1], 400)
        self.assertEqual(self.run_case(configure=lambda s, e: setattr(e, "split_engine", True))[1].calls[0][1], 400)
        self.assertEqual(self.run_case(configure=lambda s, e: setattr(e, "multi", 2))[1].calls[0][1], 400)

    def test_plain_mock_backend_is_never_capped(self):
        class RecordingMock(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, **kwargs):
                self.limit = max_new
                yield from super().generate(ids, max_new, sampling, cancel, **kwargs)
        tok = Tokenizer()
        engine = RecordingMock(tok, "abc</think>\n\nanswer")
        service = Service(engine, tok, None)
        service.reasoning_budget_engine_cap = True
        service.reasoning_budget_tokens = 20
        list(service.run(tok.encode("P:"), True, [], 400, {"seed": 17}, threading.Event()))
        self.assertEqual(engine.limit, 400)

    def test_forced_opening_at_cap_is_kept_after_wrap(self):
        force = "<tool_call>\n<function=test>"
        result = self.run_case(force=force)
        self.assertIn(0, result[1].completed)
        self.assertEqual(len(result[1].calls), 2)
        suffix = result[1].tok.encode(REASONING_WRAP_UP + force, parse_special=True)
        self.assertEqual(result[1].calls[1][0][-len(suffix):], suffix)

    def test_loaded_shared_setting_is_validated_and_can_override_config(self):
        self.assertEqual(clean_shared_defaults({"reasoning_budget_engine_cap": True}),
                         {"reasoning_budget_engine_cap": True})
        for bad in (1, "true", [], {}):
            with self.assertRaises(ValueError):
                clean_shared_defaults({"reasoning_budget_engine_cap": bad})
        off = self.run_case(configure=lambda s, e: s.set_shared({"reasoning_budget_engine_cap": False}))
        self.assertEqual(off[1].calls[0][1], 400)
        on = self.run_case(cap=False, configure=lambda s, e: s.set_shared({"reasoning_budget_engine_cap": True}))
        self.assertEqual(on[1].calls[0][1], 20)

    def test_automatic_1536_budget_and_manual_64(self):
        auto = self.run_case("x" * 3000, budget=0, maximum=2048,
                             configure=lambda s, e: setattr(s, "answer_room", True))
        self.assertEqual(auto[1].calls[0][1], 1536)
        self.assertEqual(self.run_case(budget=64)[1].calls[0][1], 64)


FAKE_BATCH = r'''
import sys
print("INFO engine=0.1.57 batch_slots=2 slot_cache=1", flush=True)
print("READY 65536 stop", flush=True)
for line in sys.stdin:
    parts = line.split()
    if not parts or parts[0] not in ("GEN", "BGEN"):
        continue
    batch = parts[0] == "BGEN"
    slot = int(parts[1]) if batch else None
    limit = int(parts[2] if batch else parts[1])
    ids = [int(x) for x in parts[-1].split(',')]
    closed = b'</think>\n\n' in bytes(x for x in ids if x < 256)
    out = list(b'Answer.' if closed else b'x' * 300) + [257]
    out = out[:limit]
    if batch:
        print('T', out[0], flush=True)
        print(f'DONE 1 {len(ids)} 1.0 2.0 length 0 0 0', flush=True)
        print(f'BADM {slot} 1', flush=True)
        for t in out[1:]: print(f'BT {slot} {t}', flush=True)
        finish = 'stop' if out[-1] == 257 else 'length'
        print(f'BDONE {slot} {len(out)} {finish} {2.0 * len(out)}', flush=True)
    else:
        for t in out: print('T', t, flush=True)
        finish = 'stop' if out[-1] == 257 else 'length'
        print(f'DONE {len(out)} {len(ids)} 1.0 {2.0 * len(out)} {finish} 0 0 0', flush=True)
'''


class NativeTransport(unittest.TestCase):
    def test_cap_drains_real_bdone_on_request_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_engine.py"
            script.write_text(f"#!{sys.executable}\n" + FAKE_BATCH)
            script.chmod(0o700)
            engine = StrataEngine(str(script), ["--batch", "2"])
            try:
                engine.waiting = 1  # force BGEN while an unrelated request waits; no HTTP or GPU
                tok = Tokenizer()
                service = Service(engine, tok, None)
                service.reasoning_budget_engine_cap = True
                service.reasoning_budget_tokens = 20
                with contextlib.redirect_stdout(io.StringIO()), mock.patch("serve.server.EOS_IDS", (257,)):
                    result = list(service.run(tok.encode("P:"), True, [], 400, {"seed": 17}, threading.Event()))
                self.assertEqual(result[-1][1]["finish"], "stop")
                self.assertEqual(service.history[-1]["engine_generated"], 28)
                self.assertEqual(service.history[-1]["decode_ms"], 56.0)
                self.assertFalse(any(engine.slot_busy))
                self.assertFalse(engine.slot_releasing)
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
