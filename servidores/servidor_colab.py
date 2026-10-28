"""TTOKEN STUDIO - servidor de video para o Google Colab (GPU).

Como usar (Colab, GPU T4): cole este arquivo inteiro em UMA celula e rode.
No fim ele imprime o LINK PUBLICO: cole esse link no cadastro do app (campo "Link do Colab").

- Modelo: SDXL-Turbo (1 passo por frame).
- Coerencia entre frames: "heranca de ruido". Cada frame parte de um ruido correlacionado com o
  anterior: z[i+1] = sqrt(1-b) * z[i] + sqrt(b) * eps  (variancia continua 1, sem passar pelo VAE).
  Seu teste mostrou que a heranca por img2img (VAE a cada frame) era 0,69x mais lenta; esta
  versao tem o mesmo custo por frame do modo independente. A coerencia (SSIM) vem na resposta
  para voce medir de verdade com `ssim_medio`.
- Seguranca: a rota /gerar exige o header  x-api-key  (padrao sk-ttoken-2026; troque pela
  variavel TTOKEN_KEY ou nos Secrets do Colab).
- Rodar em outra thread: o servidor sobe em thread e a celula do Colab continua livre.
"""
import base64
import os
import re
import subprocess
import sys
import threading
import time
from io import BytesIO

CHAVE = os.environ.get("TTOKEN_KEY", "sk-ttoken-2026")
MODELO = "stabilityai/sdxl-turbo"
PORTA = 8000
MAX_FRAMES = 16


def _garantir(pacotes):
    faltam = []
    for mod, pip in pacotes:
        try:
            __import__(mod)
        except ImportError:
            faltam.append(pip)
    if faltam:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + faltam, check=False)


# ------------------------------------------------------------------ metricas e GIF (puros)
def ssim_global(a, b):
    """SSIM simples (janela unica) entre duas imagens em tons de cinza, em float32.
    (np.uint8 - np.uint8 estoura e da numero errado: por isso o astype.)"""
    import numpy as np
    x, y = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - mx) * (y - my)).mean()
    return float(((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def ssim_medio(frames):
    """Media do SSIM entre frames consecutivos (1.0 = identicos, perto de 0 = sem relacao)."""
    if len(frames) < 2:
        return 1.0
    cinza = [f.convert("L").resize((128, 128)) for f in frames]
    vals = [ssim_global(cinza[i], cinza[i + 1]) for i in range(len(cinza) - 1)]
    return sum(vals) / len(vals)


def para_gif_b64(frames, duracao_ms=120):
    buf = BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duracao_ms, loop=0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ o motor
class MotorVideo:
    """Gera frames. `gerador(prompt, n, beta, seed)` devolve lista de imagens PIL; por padrao
    usa SDXL-Turbo na GPU. Em testes, injete um gerador falso."""

    def __init__(self, gerador=None):
        self._gerador = gerador
        self._pipe = None
        self._trava = threading.Lock()        # uma geracao por vez (a T4 e uma so)
        self.historico = []

    def _carregar(self):
        import torch
        from diffusers import AutoPipelineForText2Image
        self._torch = torch
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            MODELO, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe("aquecendo", num_inference_steps=1, guidance_scale=0.0)

    def _gerar_gpu(self, prompt, n, beta, seed):
        if self._pipe is None:
            self._carregar()
        torch = self._torch
        gen = torch.Generator("cuda").manual_seed(seed)
        lado = 512 // 8
        z = torch.randn((1, 4, lado, lado), generator=gen, device="cuda", dtype=torch.float16)
        frames = []
        for _ in range(n):
            img = self._pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0,
                             height=512, width=512, latents=z).images[0]
            frames.append(img)
            eps = torch.randn(z.shape, generator=gen, device="cuda", dtype=torch.float16)
            z = (1 - beta) ** 0.5 * z + beta ** 0.5 * eps       # heranca de ruido (variancia 1)
        return frames

    def gerar(self, prompt, frames=8, beta=0.05, seed=None):
        prompt = (prompt or "").strip()
        if len(prompt) < 3:
            raise ValueError("prompt muito curto")
        n = max(2, min(int(frames), MAX_FRAMES))
        beta = min(max(float(beta), 0.0), 1.0)
        seed = int(time.time() * 1000) % (2 ** 31) if seed is None else int(seed)
        with self._trava:
            inicio = time.time()
            fn = self._gerador or self._gerar_gpu
            imgs = fn(prompt, n, beta, seed)
            ssim = ssim_medio(imgs)
            self.historico.append(round(ssim, 3))
            return {"status": "sucesso", "frames": len(imgs), "ssim_medio": round(ssim, 3),
                    "tempo": round(time.time() - inicio, 2), "seed": seed,
                    "gif": para_gif_b64(imgs)}


# ------------------------------------------------------------------ API (FastAPI)
def criar_app(motor, chave=CHAVE):
    from typing import Optional

    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Ttoken Studio - Colab")

    class Pedido(BaseModel):
        prompt: str
        frames: int = 8
        beta: float = 0.05
        seed: Optional[int] = None

    def _checa(x_api_key):
        if x_api_key != chave:
            raise HTTPException(status_code=401, detail="chave invalida")

    @app.get("/saude")
    def saude():                                  # sem chave: o app so testa se esta no ar
        return {"status": "online", "modelo": MODELO, "ultimos_ssim": motor.historico[-5:]}

    @app.post("/gerar")
    def gerar(req: Pedido, x_api_key: str = Header(None)):   # def (nao async): nao trava o loop
        _checa(x_api_key)
        try:
            return motor.gerar(req.prompt, req.frames, req.beta, req.seed)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="falha ao gerar: %s" % str(e)[:200])

    return app


# ------------------------------------------------------------------ tunel publico
def abrir_tunel(porta=PORTA):
    """ngrok (se houver NGROK_TOKEN) ou Cloudflare Tunnel (sem cadastro). Devolve a URL https."""
    token = os.environ.get("NGROK_TOKEN", "")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("NGROK_TOKEN") or ""
        except Exception:
            token = ""
    if token:
        _garantir([("pyngrok", "pyngrok")])
        from pyngrok import ngrok
        ngrok.set_auth_token(token)
        return ngrok.connect(porta, "http").public_url
    exe = "/tmp/cloudflared"
    if not os.path.exists(exe):
        subprocess.run(["wget", "-q", "-O", exe,
                        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
                        "cloudflared-linux-amd64"], check=True)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, "tunnel", "--url", "http://localhost:%d" % porta,
                             "--no-autoupdate"], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            text=True)
    achado = []

    def le():
        for linha in proc.stderr:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", linha)
            if m and not achado:
                achado.append(m.group(0))
    threading.Thread(target=le, daemon=True).start()
    for _ in range(60):
        if achado:
            return achado[0]
        time.sleep(1)
    raise RuntimeError("o tunel nao abriu em 60 s")


def principal():
    _garantir([("diffusers", "diffusers"), ("transformers", "transformers"),
               ("accelerate", "accelerate"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
               ("PIL", "Pillow")])
    import uvicorn
    motor = MotorVideo()
    print("Carregando SDXL-Turbo na GPU (uns 40 s)...")
    motor._carregar()
    app = criar_app(motor)
    servidor = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=PORTA, log_level="warning"))
    threading.Thread(target=servidor.run, daemon=True).start()
    time.sleep(2)
    url = abrir_tunel(PORTA)
    print("\n" + "=" * 60)
    print("PRONTO. Cole este link no app (Estudio > Link do Colab):")
    print(url)
    print("Chave (x-api-key):", CHAVE)
    print("=" * 60)
    while True:                       # mantem a celula viva; interrompa para desligar
        time.sleep(60)


if __name__ == "__main__":
    principal()
