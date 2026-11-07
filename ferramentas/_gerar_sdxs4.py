"""Gera colab_sdxs_4bits.py (arquivo unico; modulos embutidos em base64+zlib, linhas curtas para colar no Colab)."""
import os, json, base64, zlib
aqui = os.path.dirname(os.path.abspath(__file__))
mods = ["cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs"]
src = {m: open(os.path.join(aqui, m + ".py"), encoding="utf-8").read() for m in mods}
b64 = base64.b64encode(zlib.compress(json.dumps(src).encode("utf-8"), 9)).decode()
emb = "_PAC = (\n" + "\n".join('    "%s"' % b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n)\n"
corpo = r'''# TTOKEN - SDXS 4 BITS: o UNet do SDXS (1 passo, sem guia) aguenta 4 bits? Testa varias receitas: SSIM/PSNR x torch fp32 + tamanho.
# Sem GPU. Me mande TODO o texto impresso + sdxs_4bits.json.
import os, sys, types, json, time, zlib, base64, subprocess, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate", "scikit-image"])
import numpy as np, torch
''' + emb + r'''
_SRC = json.loads(zlib.decompress(base64.b64decode(_PAC)).decode("utf-8"))
def _mod(nome):
    m = types.ModuleType(nome); sys.modules[nome] = m; exec(_SRC[nome], m.__dict__); return m
for _n in ("cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs"):
    _mod(_n)
cq, mc, cv, pt, us, tn, ps = [sys.modules[n] for n in ("cubo_quant", "motor_cubico", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs")]
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from diffusers import StableDiffusionPipeline, AutoencoderTiny

REPO, PROMPT = "IDKiro/sdxs-512-dreamshaper", "a red apple on a wooden table, studio photo"
PROMPTS = [PROMPT, 'a cat sitting on a sofa, photo', 'a futuristic city at night, neon lights', 'a mountain lake at sunrise, landscape photo']
pipe = StableDiffusionPipeline.from_pretrained(REPO, torch_dtype=torch.float32, safety_checker=None)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
pipe = pipe.to("cpu"); pipe.set_progress_bar_config(disable=True)
unet, te, vae, tk = pipe.unet.eval(), pipe.text_encoder.eval(), pipe.vae.eval(), pipe.tokenizer
psnr = lambda a, b: 10 * np.log10(255 ** 2 / (np.mean((a.astype(np.float64) - b) ** 2) + 1e-12))
ssim = lambda a, b: float(SSIM(np.asarray(Image.fromarray(a).convert("L")), np.asarray(Image.fromarray(b).convert("L")), data_range=255))
snr = lambda a, b: 10 * np.log10((a ** 2).mean() / (((a - b) ** 2).mean() + 1e-30))

def estado(m, tirar=None, so=None):
    d = {}
    for k, v in m.state_dict().items():
        if not v.is_floating_point(): continue
        if tirar and k.startswith(tirar): k = k[len(tirar):]
        if so and not k.startswith(so): continue
        d[k] = v.detach().float().numpy()
    return d
est_u = estado(unet); est_vae = estado(vae, so="decoder.")
cq.salvar(estado(te, "text_model."), "sdxs_texto_q8.ttq", (4, 4), 8)
cq.salvar(est_vae, "sdxs_vae_fp16.ttq", (4, 4), 8, receita={k: "fp16" for k in est_vae})
os.makedirs("tok", exist_ok=True); tk.save_pretrained("tok")
n_unet = sum(p.numel() for p in unet.parameters())
SENS = ("conv_in", "conv_out", "time_embedding", "conv_shortcut", "add_embedding")
def lin(k): return "transformer_blocks" in k or "proj_in" in k or "proj_out" in k
def rec_mista(b4):      # 4 bits so nos lineares de atencao/ff; o resto em 8 bits
    return {k: (b4, 4) if (lin(k) and not any(s in k for s in SENS)) else ((4, 4), 8) for k, v in est_u.items() if v.ndim >= 2}
def rec_sens(b4):       # tudo em 4 bits, menos as camadas sensiveis (8 bits)
    return {k: ((4, 4), 8) if any(s in k for s in SENS) else (b4, 4) for k, v in est_u.items() if v.ndim >= 2}
VARIANTES = [("8b_4x4 (base)", (4, 4), 8, None), ("4b_4x16 tudo", (4, 16), 4, None), ("4b_4x8 tudo", (4, 8), 4, None),
             ("4b_4x4 tudo", (4, 4), 4, None), ("4b_2x8 tudo", (2, 8), 4, None),
             ("4b_4x8 + sensiveis 8b", (4, 8), 4, rec_sens((4, 8))),
             ("4b so atencao/ff 4x8, resto 8b", (4, 8), 8, rec_mista((4, 8))), ("4b so atencao/ff 4x4, resto 8b", (4, 4), 8, rec_mista((4, 4)))]
LADO = 256
alvos = []
for i, p in enumerate(PROMPTS):
    lat0 = np.random.RandomState(10 + i).randn(4, LADO // 8, LADO // 8).astype(np.float32)
    with torch.no_grad():
        ref = pipe(p, num_inference_steps=1, guidance_scale=0, latents=torch.tensor(lat0)[None], height=LADO, width=LADO, output_type="np").images[0]
    alvos.append((p, lat0, (ref * 255 + 0.5).clip(0, 255).astype("uint8")))
    Image.fromarray(alvos[-1][2]).save("ref_%d.png" % i)
print("referencias torch prontas (%d prompts, %dpx)\n" % (len(alvos), LADO))
print("%-34s %8s %7s | %6s %6s" % ("receita (UNet)", "MB", "%fp16", "PSNR", "SSIM"))
resultados = []
for nome, bloco, bits, rec in VARIANTES:
    cam = "unet_tmp.ttq"
    cq.salvar(est_u, cam, bloco, bits, receita=rec)
    mb = os.path.getsize(cam) / 1e6
    d_ = "_p"; os.makedirs(d_, exist_ok=True)
    for a_, b_ in ((cam, "sdxs_unet_q8.ttq"), ("sdxs_texto_q8.ttq", "sdxs_texto_q8.ttq"), ("sdxs_vae_fp16.ttq", "sdxs_vae_fp16.ttq"), ("tok/tokenizer.json", "tokenizer.json")):
        dd = os.path.join(d_, b_)
        if os.path.lexists(dd): os.remove(dd)
        os.symlink(os.path.abspath(a_), dd)
    ps_, ss_ = [], []
    for j, (p, lat0, ref) in enumerate(alvos):
        im = ps.gerar(d_, p, lado=LADO, latente=lat0, log=lambda *_: None)
        ps_.append(psnr(ref, im)); ss_.append(ssim(ref, im))
        if nome.startswith("4b_4x8 tudo") or nome.startswith("8b"): Image.fromarray(im).save("img_%s_%d.png" % (nome.split()[0], j))
    r = {"receita": nome, "mb": mb, "pct_fp16": 100 * mb * 1e6 / (2 * n_unet), "psnr": float(np.mean(ps_)), "ssim": float(np.mean(ss_)), "ssim_min": float(np.min(ss_))}
    resultados.append(r)
    print("%-34s %8.1f %6.1f%% | %6.1f %6.3f  (pior SSIM %.3f)" % (nome, r["mb"], r["pct_fp16"], r["psnr"], r["ssim"], r["ssim_min"]))
json.dump(resultados, open("sdxs_4bits.json", "w"), indent=1)
print("\nLeitura: SSIM >= 0.95 = igual ao 8 bits; 0.90-0.95 = aceitavel; < 0.85 = estraga. Me mande TODO este texto + sdxs_4bits.json.")
try:
    from google.colab import files; files.download("sdxs_4bits.json")
except Exception:
    pass
'''
open(os.path.join(aqui, "colab_sdxs_4bits.py"), "w", encoding="utf-8").write(corpo)
print("ok", max(len(l) for l in corpo.splitlines()))
