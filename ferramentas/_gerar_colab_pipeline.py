"""Gera colab_pipeline_validar.py (arquivo unico, embute os modulos)."""
import os
aqui = os.path.dirname(os.path.abspath(__file__))
mods = ["cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd"]
src = {m: open(os.path.join(aqui, m + ".py"), encoding="utf-8").read() for m in mods}
cab = '''# TTOKEN - VALIDA O PIPELINE COMPLETO (tokenizador + agendador + UNet + CLIP + VAE em numpy) CONTRA O diffusers.
# Rode no MESMO Colab (precisa dos .ttq: tinysd_unet_q8.ttq, tinysd_texto_q8.ttq, tinysd_vae_q8.ttq; opcional tinysd_unet_curva.ttq).
# Se faltarem, rode antes o colab_tudo_validar.py. Arquivo UNICO.
import sys, types, os, time, json
import numpy as np
'''
corpo = "_SRC = {\n" + "".join("    %r: %r,\n" % (m, s) for m, s in src.items()) + "}\n"
corpo += '''
def _mod(nome):
    m = types.ModuleType(nome); sys.modules[nome] = m; exec(_SRC[nome], m.__dict__); return m
for _n in ("cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd"):
    _mod(_n)
pt = sys.modules["pipeline_tinysd"]

import torch
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32, safety_checker=None).to("cpu")
os.makedirs("tok", exist_ok=True); pipe.tokenizer.save_pretrained("tok")
PASTA = "."
tok = pt.TokenizadorCLIP("tok/tokenizer.json")

print("=== 1) TOKENIZADOR (nosso x HuggingFace) ===")
testes = ["a red apple on a table", "A photo of an astronaut riding a horse, 4k!!", "gato preto nao e azul 123 it's",
          "", "  multiple   spaces\\nnewline cafe", "a cat wearing a hat, oil painting, ultra detailed " * 8]
ok = 0
for s in testes:
    hf = pipe.tokenizer(s, padding="max_length", max_length=77, truncation=True).input_ids
    igual = tok.codificar(s) == hf; ok += igual
    print("  %-5s %r" % ("OK" if igual else "DIFF", s[:40]))
print("  tokenizador: %d/%d iguais" % (ok, len(testes)))

print("\\n=== 2) AGENDADOR (nosso x diffusers, mesmos 'eps' aleatorios) ===")
d = pipe.scheduler; d.set_timesteps(4)
a = pt.AgendadorDPM(); a.definir_passos(4)
print("  timesteps diffusers", d.timesteps.tolist(), "| nossos", a.timesteps.tolist())
r = np.random.RandomState(0); x = r.randn(4, 32, 32).astype("f"); xt = torch.tensor(x)[None]; xa = x
for k, t in enumerate(d.timesteps):
    e = r.randn(4, 32, 32).astype("f")
    xt = d.step(torch.tensor(e)[None], t, xt).prev_sample; xa = a.passo(e, xa)
    print("  passo %d erro maximo %.2e" % (k, float(abs(xt[0].numpy() - xa).max())))

print("\\n=== 3) PONTA A PONTA (mesmo prompt, mesmo ruido inicial, 4 passos, guia 7.5) ===")
prompt = "a red apple on a wooden table, studio photo"
def psnr(u, v): return 10 * np.log10(255 ** 2 / (np.mean((u.astype(np.float64) - v) ** 2) + 1e-12))
lat0 = np.random.RandomState(7).randn(4, 32, 32).astype("f")
t0 = time.time()
ref = pipe(prompt, num_inference_steps=4, guidance_scale=7.5, latents=torch.tensor(lat0)[None], height=256, width=256, output_type="np").images[0]
ref = (ref * 255 + 0.5).clip(0, 255).astype("uint8")
print("  torch fp32: %.1f s" % (time.time() - t0))
from PIL import Image
Image.fromarray(ref).save("pipe_torch.png")
unets = [f for f in ("tinysd_unet_q8.ttq", "tinysd_unet_curva.ttq", "tinysd_unet_final.ttq") if os.path.exists(f)]
for f in unets:
    # forca o arquivo escolhido: cria pasta temporaria com so esse UNet
    os.makedirs("_p", exist_ok=True)
    for n in os.listdir("_p"): os.remove(os.path.join("_p", n))
    for src_f, dst in ((f, "tinysd_unet_q8.ttq"), ("tinysd_texto_q8.ttq", "tinysd_texto_q8.ttq"), ("tinysd_vae_q8.ttq", "tinysd_vae_q8.ttq"), ("tok/tokenizer.json", "tokenizer.json")):
        os.symlink(os.path.abspath(src_f), os.path.join("_p", dst))
    t0 = time.time()
    img = pt.gerar("_p", prompt, passos=4, guia=7.5, latente=lat0, log=lambda *_: None)
    pt.salvar_png(img, "pipe_numpy_%s.png" % f.replace("tinysd_unet_", "").replace(".ttq", ""))
    print("  numpy com %-24s PSNR vs torch fp32 = %5.1f dB  (>= 35 = igual a olho) | %.0f s no Colab" % (f, psnr(ref, img), time.time() - t0))
print("\\nSalvei pipe_torch.png e pipe_numpy_*.png. Me mande este texto (e as imagens, se puder).")
'''
open(os.path.join(aqui, "colab_pipeline_validar.py"), "w", encoding="utf-8").write(cab + corpo)
print("ok")
