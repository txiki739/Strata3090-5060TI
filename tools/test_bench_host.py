"""Tests for the server address tools/lamina_bench.py polls and the port it starts a server on, without starting one.

    python -m unittest tools.test_bench_host
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lamina_bench as SB  # noqa: E402


class ServerBase(unittest.TestCase):
    def test_uses_the_configured_host(self):
        # The readiness poll was hardcoded to 127.0.0.1 while the server was started on the config's
        # host, so a LAN bind left BENCH saying "still loading" until it was killed (#2330).
        self.assertEqual(SB.server_base("192.168.1.10", 8080), "http://192.168.1.10:8080")

    def test_defaults_to_loopback_without_a_host(self):
        self.assertEqual(SB.server_base("", 9000), "http://127.0.0.1:9000")
        self.assertEqual(SB.server_base(None, 9000), "http://127.0.0.1:9000")

    def test_wildcard_binds_are_connected_through_loopback(self):
        self.assertEqual(SB.server_base("0.0.0.0", 8080), "http://127.0.0.1:8080")
        self.assertEqual(SB.server_base("::", 8080), "http://[::1]:8080")

    def test_ipv6_literals_are_bracketed(self):
        self.assertEqual(SB.server_base("fd00::5", 8080), "http://[fd00::5]:8080")
        self.assertEqual(SB.server_base("[fd00::5]", 8080), "http://[fd00::5]:8080")

    def test_a_hostname_is_left_alone(self):
        self.assertEqual(SB.server_base("bench.local", 8080), "http://bench.local:8080")


class StartingTheServer(unittest.TestCase):
    """With no server running, the bench starts one on the config's port, or on --port."""

    class Started(Exception):
        pass

    def started_on(self, *extra):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "lamina-test.json"
            cfg.write_text(json.dumps({"port": 8123, "host": "127.0.0.1"}), encoding="utf-8")
            seen = {}

            def start_server(cfg_path, port, gpu, log_path):
                seen["port"] = port
                raise self.Started

            with mock.patch.object(SB, "is_lamina", return_value=False), \
                 mock.patch.object(SB, "try_json", return_value=None), \
                 mock.patch.object(SB, "start_server", start_server), \
                 mock.patch.object(sys, "argv", ["lamina_bench.py", "--config", str(cfg), "--quiet", *extra]):
                with self.assertRaises(self.Started):
                    SB.main()
            return seen["port"]

    def test_the_configs_port(self):
        # the start path printed and passed an undefined name, so a bench that had to start the server stopped with
        # a NameError before loading anything
        self.assertEqual(self.started_on(), 8123)

    def test_the_port_option(self):
        self.assertEqual(self.started_on("--port", "8200"), 8200)


if __name__ == "__main__":
    unittest.main()
