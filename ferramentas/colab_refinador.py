# Ttoken - REFINADOR NA NUVEM (Colab, GPU T4 gratis). Cole numa celula do Colab e rode.
# O celular faz o rascunho offline; aqui so o detalhe: SDXL-Turbo img2img (2-4 passos, ~1-2 s por imagem).
# Sem token: usa o tunel gratis "cloudflared" (trycloudflare.com). Ele imprime o link no fim.
# Licenca do SDXL-Turbo: uso nao comercial/pesquisa (veja a pagina do modelo).
import subprocess, sys, os, re, time, threading, base64, io

# Guarda o modelo (7 GB) e o tunel no seu Google Drive: nas proximas vezes nao baixa de novo.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PASTA = "/content/drive/MyDrive/ttoken_refinador"
    os.makedirs(PASTA, exist_ok=True)
    os.environ["HF_HOME"] = PASTA + "/hf"
except Exception as e:
    print("Drive nao montou (%s): sigo sem guardar." % e)
    PASTA = "."

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate",
                "fastapi", "uvicorn", "pillow"], check=True)

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from fastapi import FastAPI, Request
from fastapi.responses import Response, PlainTextResponse
import uvicorn

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
app = FastAPI()
TRAVA = threading.Lock()


@app.get("/saude")
def saude():
    return PlainTextResponse("ttoken-refinador ok")


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
            print("\n=== COLE ESTE LINK NO TTOKEN (LINK DO COLAB) ===\n" + m.group(0) + "\n", flush=True)
            return


threading.Thread(target=_link, daemon=True).start()
def _servidor():                     # thread propria: o Colab ja tem um loop de eventos rodando
    import asyncio
    asyncio.run(uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")).serve())


threading.Thread(target=_servidor, daemon=True).start()
print("Refinador no ar. Deixe esta celula rodando.", flush=True)
while True:
    time.sleep(60)
