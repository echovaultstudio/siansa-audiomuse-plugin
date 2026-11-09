"""The page's script must parse: one mistake there (a name declared twice) leaves the whole page empty."""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

PAGE = os.path.join(os.path.dirname(__file__), "..", "beats", "page.html")


class PageScriptTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_the_page_script_parses(self):
        script = "\n".join(re.findall(r"<script>(.*?)</script>", open(PAGE, encoding="utf-8").read(), re.S))
        self.assertTrue(script.strip())
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(script)
        try:
            check = subprocess.run(["node", "--check", f.name], capture_output=True, text=True)
        finally:
            os.unlink(f.name)
        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
