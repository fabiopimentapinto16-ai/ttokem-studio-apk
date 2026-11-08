"""Gera sdxs_servidor.py: motor rapido para o Pydroid 3. Fica rodando e o app Ttoken pede as imagens (127.0.0.1:8765)."""
import os, json, base64, zlib
aqui = os.path.dirname(os.path.abspath(__file__))
mods = ["cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs"]
src = {m: open(os.path.join(aqui, m + ".py"), encoding="utf-8").read() for m in mods}
b64 = base64.b64encode(zlib.compress(json.dumps(src).encode("utf-8"), 9)).decode()
emb = "_PAC = (\n" + "\n".join('    "%s"' % b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n)\n"
corpo = r'''# TTOKEN - MOTOR RAPIDO (Pydroid 3). Rode este arquivo UMA vez e deixe o Pydroid aberto: o app Ttoken pede as imagens aqui
# (45 s por imagem em vez de ~12 min dentro do app). Precisa da pasta sdxs_cubo em Download. So atende este celular (127.0.0.1).
PASTA = ""            # vazio = acha sozinho a pasta COMPLETA (sdxs_cubo)
PORTA = 8765
import os, sys, types, json, time, zlib, base64, threading, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
''' + emb + r'''
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
'''
open(os.path.join(aqui, "sdxs_servidor.py"), "w", encoding="utf-8").write(corpo)
print("ok", max(len(l) for l in corpo.splitlines()))
