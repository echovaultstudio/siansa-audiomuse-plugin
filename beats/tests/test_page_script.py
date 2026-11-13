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

    def test_the_menu_and_the_pages_match_and_nothing_old_is_said(self):
        """0.43 (the owner, 13 Nov: pages still spoke of things since removed): every menu entry has its page and the
        script knows it; and the page says nothing of AudioMuse-AI or a plugin, which Siansa Beats no longer has."""
        page = open(PAGE, encoding="utf-8").read()
        menu = re.findall(r'<a href="#(\w+)" data-view="(\w+)"', page)
        self.assertEqual([a for a, _ in menu], [b for _, b in menu])
        sections = set(re.findall(r'<section data-view="(\w+)"', page))
        views = set(re.search(r'const VIEWS = \[(.*?)\];', page).group(1).replace('"', "").replace(" ", "").split(","))
        self.assertEqual({b for _, b in menu} - sections, set())
        self.assertEqual(sections, views)
        self.assertEqual([b for _, b in menu], ["overview", "libraries", "songs", "listening", "loudness", "keys", "genres",
                                                "similar", "activity", "setup", "log", "blends", "report"])
        for gone in ("audiomuse", "plugin"):
            self.assertNotIn(gone, page.lower(), gone)
        signin = open(os.path.join(os.path.dirname(PAGE), "signin.html"), encoding="utf-8").read()
        self.assertIn("beats.reset", signin)  # the way back in is said where it's needed
        self.assertNotIn("audiomuse", signin.lower())


if __name__ == "__main__":
    unittest.main()
