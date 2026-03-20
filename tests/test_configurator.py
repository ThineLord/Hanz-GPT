import pathlib
import os
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGURATOR_CODE = (ROOT / "configurator.py").read_text(encoding="utf-8")


def run_configurator(argv, initial_globals):
    scope = dict(initial_globals)
    scope["__name__"] = "__main__"
    old_argv = sys.argv[:]
    try:
        sys.argv = ["tool.py", *argv]
        exec(CONFIGURATOR_CODE, scope)
    finally:
        sys.argv = old_argv
    return scope


class ConfiguratorSmokeTest(unittest.TestCase):
    def test_key_value_with_equal_sign_in_value(self):
        scope = run_configurator(["--start=A=B"], {"start": "\n"})
        self.assertEqual(scope["start"], "A=B")

    def test_config_file_then_cli_override(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("batch_size = 8\n")
            config_path = f.name
        try:
            scope = run_configurator([config_path, "--batch_size=16"], {"batch_size": 4})
            self.assertEqual(scope["batch_size"], 16)
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
