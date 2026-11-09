import base64, json, os, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nuvem

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 20


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ttoken-refinador ok")
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert d["prompt"] and base64.b64decode(d["imagem_b64"])
        self.send_response(200); self.send_header("Content-Type", "image/png"); self.end_headers(); self.wfile.write(PNG)


class T(unittest.TestCase):
    def test_refinar_e_fallback(self):
        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d/" % srv.server_port
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            open(p, "wb").write(PNG)
            self.assertTrue(nuvem.pronto(url))
            out = nuvem.refinar(p, "a cat", url)
            self.assertTrue(out and os.path.exists(out))
            self.assertIsNone(nuvem.refinar(p, "a cat", "http://127.0.0.1:1"))
            self.assertIsNone(nuvem.refinar(p, "a cat", ""))
        srv.shutdown()


if __name__ == "__main__":
    unittest.main()
