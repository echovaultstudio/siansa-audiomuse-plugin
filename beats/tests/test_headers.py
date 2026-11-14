"""A header's value never carries a line break: one would start a header of its own (response splitting)."""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from beats.server import Handler  # noqa: E402


class HeadersTest(unittest.TestCase):
    def test_a_line_break_in_a_header_value_is_taken_out(self):
        h = object.__new__(Handler)
        h.wfile = io.BytesIO()
        h.headers = {}
        h.command = "GET"
        h.request_version = "HTTP/1.1"
        h.requestline = "GET / HTTP/1.1"
        h.client_address = ("127.0.0.1", 0)
        h.log_request = lambda *a, **k: None
        h._send(200, b"x", kind="image/png\r\nSet-Cookie: a=b", headers={"X-Name": "one\r\nX-Evil: two"})
        head = h.wfile.getvalue().split(b"\r\n\r\n")[0].decode()
        lines = head.split("\r\n")
        self.assertFalse([line for line in lines if line.startswith(("Set-Cookie", "X-Evil"))], lines)
        self.assertIn("Content-Type: image/png  Set-Cookie: a=b", lines)


if __name__ == "__main__":
    unittest.main()
