#!/usr/bin/env python3
"""CPU/source checks for the API serve first-logits diagnostic."""
from pathlib import Path
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/program/generate.cpp").read_text()
MARKER = "                // The API uses this serve loop, not the standalone generation loop's diagnostic hook."
START = SOURCE.index(MARKER)
END = SOURCE.index("                int a = 0;", START)
HOOK = SOURCE[START:END]


class FirstLogitsDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("A C++ compiler is required for diagnostic CPU verification")
        cls.directory = tempfile.TemporaryDirectory()
        cls.base = Path(cls.directory.name)
        unit = cls.base / "diagnostic.cpp"
        unit.write_text("""#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
struct Verifier {
    bool copy_ok; int count;
    int vocab() const { return count; }
    bool copy_logits(int t, float* row) const {
        if (!copy_ok || t != 0) return false;
        const float expected[] = {0.25f, -1.5f, 3.125f, 7.0f};
        for (int i = 0; i < count; ++i) row[i] = expected[i % 4];
        return true;
    }
};
int run(bool first_window, bool copy_ok, int count) {
    Verifier ver{copy_ok, count}; std::string err;
""" + HOOK + """
    return 0;
}
int main(int argc, char** argv) {
    return run(std::atoi(argv[1]) != 0, std::atoi(argv[2]) != 0, std::atoi(argv[3]));
}
""")
        cls.binary = cls.base / "diagnostic"
        subprocess.run([compiler, "-std=c++17", str(unit), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def execute(self, path=None, first=True, copy=True, count=4):
        env = dict(os.environ)
        env.pop("STRATA_DUMP_FIRST_LOGITS", None)
        if path is not None:
            env["STRATA_DUMP_FIRST_LOGITS"] = str(path)
        return subprocess.run([str(self.binary), str(int(first)), str(int(copy)), str(count)], env=env,
                              capture_output=True, text=True)

    def test_hook_is_in_serve_decode_after_run_before_commit(self):
        serve = SOURCE.index("    if (o.serve) {")
        self.assertLess(serve, START)
        self.assertLess(START, SOURCE.index('        save_profile("exit");', serve))
        run = SOURCE.rfind("if (!ver.run(", serve, START)
        self.assertGreater(run, serve)
        self.assertNotIn("ver.commit(", SOURCE[run:START])
        self.assertLess(END, SOURCE.index("if (!ver.commit(", END))
        self.assertIn("if (first_window)", HOOK)
        self.assertNotIn("first_window =", HOOK)

    def test_first_window_writes_exact_raw_full_row(self):
        path = self.base / "row.f32"
        result = self.execute(path)
        self.assertEqual((result.returncode, result.stdout), (0, ""))
        self.assertEqual(path.read_bytes(), struct.pack("=4f", 0.25, -1.5, 3.125, 7.0))

    def test_later_window_and_unarmed_request_do_nothing(self):
        path = self.base / "later.f32"
        self.assertEqual(self.execute(path, first=False, copy=False).returncode, 0)
        self.assertFalse(path.exists())
        self.assertEqual(self.execute(copy=False).returncode, 0)

    def test_copy_failure_stops_without_creating_file(self):
        path = self.base / "copy-failed.f32"
        result = self.execute(path, copy=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "ERR first logits copy failed\n")
        self.assertFalse(path.exists())

    def test_open_failure_stops_request(self):
        result = self.execute(self.base / "missing" / "row.f32")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ERR first logits open failed:", result.stdout)

    @unittest.skipUnless(Path("/dev/full").exists(), "/dev/full exercises real output failure")
    def test_write_failure_stops_request_and_closes_file(self):
        result = self.execute("/dev/full", count=20000)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ERR first logits write failed:", result.stdout)

    @unittest.skipUnless(Path("/dev/full").exists(), "/dev/full exercises buffered close failure")
    def test_close_failure_stops_request(self):
        result = self.execute("/dev/full")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ERR first logits close failed:", result.stdout)


if __name__ == "__main__":
    unittest.main()
