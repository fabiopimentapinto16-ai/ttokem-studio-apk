# TTOKEN - MOTOR RAPIDO (Pydroid 3). Rode este arquivo UMA vez e deixe o Pydroid aberto: o app Ttoken pede as imagens aqui
# (45 s por imagem em vez de ~12 min dentro do app). Precisa da pasta sdxs_cubo em Download. So atende este celular (127.0.0.1).
PASTA = ""            # vazio = acha sozinho a pasta COMPLETA (sdxs_cubo)
PORTA = 8765
import os, sys, types, json, time, zlib, base64, threading, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
_PAC = (
    "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"
)

_SRC = json.loads(zlib.decompress(base64.b64decode(_PAC)).decode("utf-8"))
def _mod(nome):
    m = types.ModuleType(nome); sys.modules[nome] = m; exec(_SRC[nome], m.__dict__); return m
for _n in ("cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs"):
    _mod(_n)
pt, ps = sys.modules["pipeline_tinysd"], sys.modules["pipeline_sdxs"]

def completo(d):
    f = lambda n: os.path.exists(os.path.join(d, n))
    return f("sdxs_unet_q8.ttq") and f("sdxs_texto_q8.ttq") and f("sdxs_vae_fp16.ttq") and (f("tokenizer.json") or f(os.path.join("tok", "tokenizer.json")))
def achar():
    for c in ([PASTA] if PASTA else []) + ["/storage/emulated/0/Download/sdxs_cubo", "/sdcard/Download/sdxs_cubo", os.getcwd()]:
        if c and completo(c): return c
    for r in ("/storage/emulated/0", "/sdcard"):
        base = r.count("/")
        for d, subs, arqs in os.walk(r):
            if d.count("/") - base >= 4: subs[:] = []
            if "sdxs_unet_q8.ttq" in arqs and completo(d): return d
    raise SystemExit("Nao achei a pasta COMPLETA sdxs_cubo (3 .ttq + tokenizer.json). Ajuste PASTA no topo.")
pasta = achar(); print("pasta dos modelos:", pasta)
TRAVA = threading.Lock()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _txt(self, cod, txt, tipo="text/plain; charset=utf-8"):
        d = txt if isinstance(txt, bytes) else txt.encode("utf-8")
        self.send_response(cod); self.send_header("Content-Type", tipo); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self):
        self._txt(200, "ttoken-sdxs ok")
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0")); req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            prompt = str(req.get("prompt", ""))[:300]; lado = int(req.get("lado", 256)); lado = 512 if lado >= 512 else 256
            semente = int(req.get("semente", 1))
            print("[%s] pedido: %r (%dpx)" % (time.strftime("%H:%M:%S"), prompt, lado), flush=True)
            with TRAVA:
                t0 = time.time()
                img = ps.gerar(pasta, prompt, semente=semente, lado=lado, log=lambda s: print("   ", s, flush=True))
                tmp = os.path.join(tempfile.gettempdir(), "ttk_sdxs_%d.png" % os.getpid()); pt.salvar_png(img, tmp)
                dados = open(tmp, "rb").read()
            print("   pronto em %.1f s" % (time.time() - t0), flush=True)
            self._txt(200, dados, "image/png")
        except Exception as e:
            print("   ERRO:", e, flush=True); self._txt(500, "erro: %s" % e)

print("\nMOTOR RAPIDO LIGADO em 127.0.0.1:%d. Deixe o Pydroid aberto e volte ao Ttoken Studio." % PORTA, flush=True)
ThreadingHTTPServer(("127.0.0.1", PORTA), H).serve_forever()
