# Ttoken - REFINADOR NA NUVEM (Colab, GPU T4 gratis). Cole numa celula do Colab e rode.
# Gera a imagem inteira (/gerar) ou refina um rascunho (/refinar) com SDXL-Turbo: ~1-2 s por imagem na GPU T4.
# Sem token: usa o tunel gratis "cloudflared" (trycloudflare.com). Ele imprime o link no fim.
# Licenca do SDXL-Turbo: uso nao comercial/pesquisa (veja a pagina do modelo).
import subprocess, sys, os, re, time, threading, base64, io

# O modelo (~7 GB) baixa no disco do Colab (rapido, ~1-2 min). NAO vai para o Drive: ele nao cabe na maioria das contas.
# Aqui so limpamos o que ficou la de tentativas anteriores, para liberar espaco, e guardamos o tunel (pequeno).
PASTA = "."
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PASTA = "/content/drive/MyDrive/ttoken_refinador"
    os.makedirs(PASTA, exist_ok=True)
    import shutil as _sh
    for velho in ("hf_modelo", "hf"):
        _sh.rmtree(PASTA + "/" + velho, ignore_errors=True)
except Exception as e:
    print("Drive nao montou (%s): sigo sem ele." % e)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate",
                "fastapi", "uvicorn", "pillow"], check=True)

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from fastapi import FastAPI, Request
from fastapi.responses import Response, PlainTextResponse
import uvicorn

def _carregar(**kw):
    return AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,
                                                     variant="fp16", cache_dir="/content/hf_local", **kw).to("cuda")


try:
    pipe_t = _carregar()
except Exception as e:                                   # download incompleto: apaga e baixa de novo
    print("Modelo com problema (%s). Baixando de novo..." % str(e)[:80], flush=True)
    import shutil
    shutil.rmtree("/content/hf_local", ignore_errors=True)
    pipe_t = _carregar(force_download=True)
pipe = AutoPipelineForImage2Image.from_pipe(pipe_t)          # reaproveita os mesmos pesos (sem gastar mais memoria)
app = FastAPI()
TRAVA = threading.Lock()


@app.get("/saude")
def saude():
    return PlainTextResponse("ttoken-refinador ok")


@app.post("/gerar")
async def gerar(req: Request):
    """Texto -> imagem inteira (SDXL-Turbo, 512 px, 4 passos, ~1-2 s numa T4)."""
    d = await req.json()
    sem = d.get("semente")
    g = torch.Generator("cuda").manual_seed(int(sem)) if sem is not None else None
    with TRAVA:
        out = pipe_t(prompt=str(d.get("prompt", ""))[:300], num_inference_steps=4, guidance_scale=0.0,
                     width=512, height=512, generator=g).images[0]
    b = io.BytesIO(); out.save(b, "PNG")
    return Response(b.getvalue(), media_type="image/png")


@app.post("/refinar")
async def refinar(req: Request):
    d = await req.json()
    img = Image.open(io.BytesIO(base64.b64decode(d["imagem_b64"]))).convert("RGB").resize((512, 512), Image.LANCZOS)
    forca = float(d.get("forca", 0.7))
    forca = min(max(forca, 0.3), 0.95)
    passos = 4 if forca < 0.6 else 3          # SDXL-Turbo faz int(passos*forca) passos; precisa >= 1
    with TRAVA:
        out = pipe(prompt=str(d.get("prompt", ""))[:300], image=img, strength=forca, num_inference_steps=passos,
                   guidance_scale=0.0).images[0]
    b = io.BytesIO(); out.save(b, "PNG")
    return Response(b.getvalue(), media_type="image/png")


CF = PASTA + "/cloudflared"
if not os.path.exists(CF):
    subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O %s && chmod +x %s" % (CF, CF), shell=True, check=True)
tun = subprocess.Popen([CF, "tunnel", "--url", "http://localhost:8000"], stderr=subprocess.PIPE, text=True)


def _link():
    for ln in tun.stderr:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", ln)
        if m:
            print("\n=== LINK DO REFINADOR (o Ttoken acha sozinho se o segredo estiver no Colab) ===\n" + m.group(0) + "\n", flush=True)
            _avisar_backend(m.group(0))
            return


API = "https://ttokem-backend-8ba2a.containers.snapdeploy.app"


def _avisar_backend(url):
    """Registra o link no backend (e reenvia a cada 5 min) para o app achar sozinho. Precisa do segredo
    COLAB_SECRET em Colab > chave (Secrets) e igual ao COLAB_SECRET do backend. Nunca cole o segredo no codigo."""
    try:
        from google.colab import userdata
        segredo = userdata.get("COLAB_SECRET")
    except Exception:
        segredo = ""
    if not segredo:
        print("Sem COLAB_SECRET nos Secrets do Colab: cole o link acima no app (LINK DO COLAB).", flush=True)
        return
    import requests

    def _loop():
        while True:
            try:
                r = requests.post(API + "/api/refinador/registrar", json={"url": url, "secret": segredo}, timeout=60)
                print("registro no backend:", r.status_code, flush=True) if r.status_code != 200 else None
            except Exception as e:
                print("registro no backend falhou:", str(e)[:80], flush=True)
            time.sleep(300)
    threading.Thread(target=_loop, daemon=True).start()


threading.Thread(target=_link, daemon=True).start()
def _servidor():                     # thread propria: o Colab ja tem um loop de eventos rodando
    import asyncio
    asyncio.run(uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")).serve())


threading.Thread(target=_servidor, daemon=True).start()
print("Refinador no ar. Deixe esta celula rodando.", flush=True)
while True:
    time.sleep(60)
