"""Tests for setup.py's --vision gpu[:GPU]: the GPU it picks, the config's `vision` section and the encoder's build,
without building or running anything.

    python -m unittest tools.test_setup_vision
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

MAIN = setup.Gpu(0, "NVIDIA GeForce RTX 3090", "GPU-aaaa", 24576, 86, False)
TIER = setup.Gpu(1, "NVIDIA GeForce RTX 3090", "GPU-bbbb", 24576, 86, True)
SMALL = setup.Gpu(2, "NVIDIA GeForce RTX 5060 Ti", "GPU-cccc", 16311, 120, False)
ALL = [MAIN, TIER, SMALL]


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class PickVisionGpu(unittest.TestCase):
    def test_the_one_named(self):
        self.assertEqual(setup.pick_vision_gpu("1", ALL, MAIN), TIER)
        self.assertEqual(setup.pick_vision_gpu("GPU-cccc", ALL, MAIN), SMALL)
        self.assertEqual(setup.pick_vision_gpu("5060", ALL, MAIN), SMALL)
        self.assertEqual(setup.pick_vision_gpu("0", ALL, MAIN), MAIN)   # the main GPU when asked for

    def test_a_name_no_gpu_has(self):
        self.assertIsNone(setup.pick_vision_gpu("4090", ALL, MAIN))

    def test_default_the_least_vram_besides_the_main_gpu(self):
        self.assertEqual(setup.pick_vision_gpu(None, ALL, MAIN), SMALL)
        self.assertEqual(setup.pick_vision_gpu(None, [MAIN, TIER], MAIN), TIER)

    def test_default_on_one_gpu_is_that_gpu(self):
        self.assertEqual(setup.pick_vision_gpu(None, [MAIN], MAIN), MAIN)


class VisionSection(unittest.TestCase):
    EXAMPLE = {"exe": "./build-vision/bin/lamina-vision", "mmproj": "m.gguf", "model": "x.gguf", "gpu": False,
               "threads": 16, "max_tokens": 1024}

    def test_the_cpu_encoder(self):
        v = setup.vision_section(self.EXAMPLE, None)
        self.assertEqual(v["exe"], "./build-vision/bin/lamina-vision")
        self.assertEqual(v["threads"], os.cpu_count() or 8)
        self.assertFalse(v["gpu"])
        self.assertNotIn("cuda_device", v)

    def test_the_cuda_encoder_on_a_card(self):
        v = setup.vision_section(self.EXAMPLE, SMALL)
        self.assertEqual(v["exe"], "./build-vision-cuda/bin/lamina-vision")
        self.assertIs(v["gpu"], True)
        self.assertEqual(v["cuda_device"], "GPU-cccc")
        self.assertNotIn("threads", v)
        self.assertEqual((v["mmproj"], v["model"], v["max_tokens"]), ("m.gguf", "x.gguf", 1024))

    def test_the_example_is_not_changed(self):
        before = dict(self.EXAMPLE)
        setup.vision_section(self.EXAMPLE, SMALL)
        self.assertEqual(self.EXAMPLE, before)


class WriteConfig(unittest.TestCase):
    def written(self, vision_gpu):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            (Path(d) / "examples").mkdir()
            shutil.copy(ROOT / "examples" / "ud-q4_k_xl.json", Path(d) / "examples")
            path = setup.write_config("ud-q4_k_xl", Path(d) / "data", "NVIDIA GeForce RTX 3090", 8082, True, None, None,
                                      vision_gpu)
            return json.loads(path.read_text(encoding="utf-8"))

    def test_the_examples_section_on_a_gpu(self):
        v = self.written(SMALL)["vision"]
        self.assertTrue(v["exe"].startswith("./build-vision-cuda/bin/lamina-vision"))
        self.assertEqual((v["gpu"], v["cuda_device"]), (True, "GPU-cccc"))
        self.assertNotIn("threads", v)

    def test_without_a_gpu_as_before(self):
        v = self.written(None)["vision"]
        self.assertTrue(v["exe"].startswith("./build-vision/bin/lamina-vision"))
        self.assertEqual((v["gpu"], v["threads"]), (False, os.cpu_count() or 8))


class BuildVision(unittest.TestCase):
    TC = setup.Toolchain(None, None, "/opt/cuda/bin/nvcc", (13, 4), None, None)

    def test_the_cpu_build_as_before(self):
        with mock.patch.object(setup, "cmake_build") as cb:
            quiet(setup.build_vision, self.TC, {})
        _, src, b, target, defs, _ = cb.call_args.args
        self.assertEqual((src, b, target), (setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "lamina-vision"))
        self.assertIn("-DLAMINA_VISION_CUDA=OFF", defs)

    def test_the_cuda_build(self):
        with mock.patch.object(setup, "cmake_build") as cb, \
                mock.patch.object(setup, "cuda_defs", return_value=["-DCMAKE_CUDA_COMPILER=nvcc"]), \
                mock.patch.object(setup, "WIN", False):
            quiet(setup.build_vision, self.TC, {}, "86;120")
        _, src, b, target, defs, _ = cb.call_args.args
        self.assertEqual((b, target), (setup.ROOT / "build-vision-cuda", "lamina-vision"))
        for d in ("-DLAMINA_VISION_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86;120", "-DCMAKE_CUDA_COMPILER=nvcc"):
            self.assertIn(d, defs)


if __name__ == "__main__":
    unittest.main()
