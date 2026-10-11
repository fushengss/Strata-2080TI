"""Setup's recommendations from the community bench data (0.1.40): text only, never a changed default.

    python -m unittest tools.test_setup_bench_tips
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_hybrid import fake_sys  # noqa: E402

BASE = ["--expert-cache", "auto", "--prefill", "auto", "--max-context", "65536"]


def tips(args=BASE, env=None, ram=32.0, model=20.0, vram=12.0, vision="none", win=False, gpus=1):
    return setup.bench_tips(list(args), env, ram, model, vram, vision, win, gpus)


class BenchTips(unittest.TestCase):
    def test_nothing_to_say_on_a_plain_pc(self):
        self.assertEqual(tips(), [])

    def test_explicit_cache_warns_on_windows_only(self):
        a = ["--expert-cache", "2400", "--prefill", "auto"]
        self.assertTrue(any("--expert-cache 2400" in t and "7x" in t for t in tips(a, win=True)))
        self.assertEqual(tips(a, win=False), [])
        self.assertEqual(tips(BASE, win=True), [])                      # auto: nothing to warn about

    def test_prefill_auto_32768_follows_the_ram(self):
        self.assertTrue(any("auto:32768" in t and t.startswith("tip") for t in tips(ram=128)))
        self.assertEqual([t for t in tips(ram=64) if "auto:32768" in t], [])
        big = ["--prefill", "auto:32768"]
        self.assertTrue(any(t.startswith("warning") and "auto:32768" in t for t in tips(big, ram=32)))
        self.assertEqual([t for t in tips(big, ram=96) if "auto:32768" in t], [])

    def test_big_prefill_chunks_on_a_layer_split(self):
        # 8x RTX 2080 Ti, 377 GB: no auto:32768 tip on a split, and a warning when the config already has it
        self.assertEqual([t for t in tips(ram=377, gpus=8) if "auto:32768" in t], [])
        for big in ("auto:32768", "auto:16384"):
            w = [t for t in tips(["--prefill", big], ram=377, gpus=8) if big in t]
            self.assertEqual(len(w), 1)
            self.assertTrue(w[0].startswith("warning") and "8 GPUs" in w[0] and "--prefill auto " in w[0])
        self.assertEqual([t for t in tips(["--prefill", "auto:32768"], ram=377, gpus=1) if "auto:32768" in t], [])

    def test_split_chunks_on_a_big_split(self):
        self.assertTrue(any("--split-chunks 4" in t and t.startswith("tip") for t in tips(ram=377, gpus=8)))
        self.assertEqual([t for t in tips(ram=377, gpus=2) if "--split-chunks" in t], [])
        self.assertEqual([t for t in tips(ram=377, gpus=1) if "--split-chunks" in t], [])
        self.assertEqual([t for t in tips([*BASE, "--split-chunks", "4"], ram=377, gpus=8) if "--split-chunks" in t], [])

    def test_resident_headroom_on_small_ram(self):
        res = [*BASE, "--resident-experts"]
        self.assertTrue(any("STRATA_RESIDENT_HEADROOM_GIB" in t for t in tips(res, ram=32)))
        self.assertTrue(any("STRATA_RESIDENT_HEADROOM_GIB" in t for t in tips([*BASE, "--resident-budget-gib", "40"], ram=48)))
        self.assertEqual([t for t in tips(res, ram=64) if "HEADROOM" in t], [])
        self.assertEqual([t for t in tips(res, {"STRATA_RESIDENT_HEADROOM_GIB": "8"}, ram=32) if "HEADROOM" in t], [])
        self.assertEqual([t for t in tips(BASE, ram=32) if "HEADROOM" in t], [])   # not a resident config

    def test_conversation_cache_for_agents(self):
        self.assertTrue(any("--conversation-cache-mib 8192" in t for t in tips(ram=64, model=40)))
        self.assertEqual([t for t in tips(ram=64, model=41) if "conversation" in t], [])      # < 24 GB left
        self.assertEqual([t for t in tips([*BASE, "--conversation-cache-mib", "4096"], ram=128) if "conversation" in t], [])

    def test_vision_cpu_on_small_cards(self):
        self.assertTrue(any("--vision cpu" in t for t in tips(vision="gpu", vram=12)))
        self.assertEqual([t for t in tips(vision="gpu", vram=24) if "vision" in t], [])
        self.assertEqual([t for t in tips(vision="cpu", vram=12) if "vision" in t], [])

    def test_tips_never_touch_the_arguments(self):
        a = ["--expert-cache", "2400", "--prefill", "auto", "--resident-experts"]
        before = list(a)
        env = {}
        tips(a, env, ram=128, vision="gpu", win=True)
        self.assertEqual(a, before)
        self.assertEqual(env, {})


class Sockets(unittest.TestCase):
    def fake(self, d, sockets, cores):
        n = 0
        for s in range(sockets):
            for c in range(cores):
                t = Path(d) / "devices" / "system" / "cpu" / f"cpu{n}" / "topology"
                t.mkdir(parents=True)
                (t / "physical_package_id").write_text(str(s))
                (t / "core_id").write_text(str(c))
                n += 1

    def test_linux_sockets(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake(d, 2, 18)
            self.assertEqual(setup.linux_sockets(d), (2, 18))
        with tempfile.TemporaryDirectory() as d:
            self.fake(d, 1, 8)
            self.assertEqual(setup.linux_sockets(d), (1, 8))
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.linux_sockets(d))

    def test_two_socket_tip(self):
        t = setup.two_socket_note((2, 18))
        self.assertEqual(len(t), 1)
        self.assertIn("--pool-workers 17", t[0])
        self.assertEqual(setup.two_socket_note((1, 18)), [])
        self.assertEqual(setup.two_socket_note(None), [])


if __name__ == "__main__":
    unittest.main()
