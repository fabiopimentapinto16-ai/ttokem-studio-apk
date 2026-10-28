"""TTOKEN STUDIO - servidor LOCAL de reserva (Pydroid 3, no proprio celular).

Sem GPU no celular: usa SD-Turbo (nao o XL) em CPU, 256x256, poucos frames. Isso e lento
(dezenas de segundos por imagem) e so serve como ultimo recurso antes das APIs gratis. Se o
torch/diffusers nao estiverem instalados, ou o servidor nao estiver rodando, o app recebe
"Offline" e passa para o Colab.

Pydroid 3: instale (Pip) torch, diffusers, transformers, accelerate, pillow, numpy e rode este
arquivo. Ele imprime  http://127.0.0.1:8000#chave : cole isso no cadastro do app.
Usa so a biblioteca padrao para o servidor HTTP (nao precisa de FastAPI).
"""
import base64
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

CHAVE = os.environ.get("TTOKEN_KEY") or secrets.token_urlsafe(9)   # aleatoria, so sua
MODELO = "stabilityai/sd-turbo"
PORTA = 8000
LADO = 256
MAX_FRAMES = 6


class MotorLocal:
    def __init__(self, gerador=None):
        self._gerador, self._pipe, self._trava = gerador, None, threading.Lock()

    def disponivel(self):
        if self._gerador:
            return True
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def _gerar_cpu(self, prompt, n, beta, seed):
        import torch
        if self._pipe is None:
            from diffusers import AutoPipelineForText2Image
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                MODELO, torch_dtype=torch.float32)
            self._pipe.set_progress_bar_config(disable=True)
        gen = torch.Generator("cpu").manual_seed(seed)
        z = torch.randn((1, 4, LADO // 8, LADO // 8), generator=gen)
        frames = []
        for _ in range(n):
            frames.append(self._pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0,
                                     height=LADO, width=LADO, latents=z).images[0])
            z = (1 - beta) ** 0.5 * z + beta ** 0.5 * torch.randn(z.shape, generator=gen)
        return frames

    def gerar(self, prompt, frames=4, beta=0.05):
        n = max(2, min(int(frames), MAX_FRAMES))
        seed = int(time.time() * 1000) % (2 ** 31)
        with self._trava:
            inicio = time.time()
            imgs = (self._gerador or self._gerar_cpu)(prompt, n, float(beta), seed)
            buf = BytesIO()
            imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                         duration=200, loop=0)
            return {"status": "sucesso", "frames": len(imgs), "tempo": round(time.time() - inicio, 2),
                    "gif": base64.b64encode(buf.getvalue()).decode("ascii")}


def fazer_handler(motor, chave=CHAVE):
    class H(BaseHTTPRequestHandler):
        def _json(self, codigo, obj):
            corpo = json.dumps(obj).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def do_GET(self):
            if self.path.rstrip("/") == "/saude":
                if motor.disponivel():
                    return self._json(200, {"status": "online", "modelo": MODELO})
                return self._json(503, {"erro": "Offline: instale torch e diffusers"})
            self._json(404, {"erro": "rota inexistente"})

        def do_POST(self):
            if self.path.rstrip("/") != "/gerar":
                return self._json(404, {"erro": "rota inexistente"})
            if self.headers.get("x-api-key") != chave:
                return self._json(401, {"erro": "chave invalida"})
            if not motor.disponivel():
                return self._json(503, {"erro": "Offline: instale torch e diffusers"})
            try:
                tam = int(self.headers.get("Content-Length") or 0)
                pedido = json.loads(self.rfile.read(tam) or b"{}")
                prompt = (pedido.get("prompt") or "").strip()
                if len(prompt) < 3:
                    return self._json(400, {"erro": "prompt muito curto"})
                self._json(200, motor.gerar(prompt, pedido.get("frames", 4),
                                            pedido.get("beta", 0.05)))
            except Exception as e:
                self._json(500, {"erro": "falha ao gerar: %s" % str(e)[:200]})

        def log_message(self, *a):          # sem barulho no console do Pydroid
            pass
    return H


def iniciar_em_thread(chave, porta=PORTA, motor=None):
    """Liga o servidor dentro do proprio app (thread). Levanta OSError se a porta estiver
    ocupada (ex.: o Pydroid ja esta rodando o servidor)."""
    srv = ThreadingHTTPServer(("127.0.0.1", porta), fazer_handler(motor or MotorLocal(), chave))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def principal():
    motor = MotorLocal()
    print("Servidor local Ttoken (modelo disponivel: %s)" % motor.disponivel())
    print("Cole no app:  http://127.0.0.1:%d#%s" % (PORTA, CHAVE))
    ThreadingHTTPServer(("127.0.0.1", PORTA), fazer_handler(motor, CHAVE)).serve_forever()


if __name__ == "__main__":
    principal()
