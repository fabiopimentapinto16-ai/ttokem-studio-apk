"""Gera colab_sdxs_validar.py (arquivo unico; modulos embutidos em base64+zlib, linhas curtas para colar no Colab)."""
import os, json, base64, zlib
aqui = os.path.dirname(os.path.abspath(__file__))
mods = ["cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs"]
src = {m: open(os.path.join(aqui, m + ".py"), encoding="utf-8").read() for m in mods}
b64 = base64.b64encode(zlib.compress(json.dumps(src).encode("utf-8"), 9)).decode()
emb = "_PAC = (\n" + "\n".join('    "%s"' % b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n)\n"
corpo = r'''# TTOKEN - SDXS-512 (1 PASSO, SEM GUIA): exporta os pesos em .ttq e VALIDA o motor numpy contra o torch. Arquivo unico.
# Mede: texto (numpy x torch), UNet (8 bits x fp32), TAESD, imagem completa em 256 e 512 px (PSNR/SSIM) e tempo no Colab.
# Gera: sdxs_unet_q8.ttq, sdxs_texto_q8.ttq, sdxs_vae_fp16.ttq, tok/tokenizer.json (+ baixa um .zip com tudo).
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
BLOCO, BITS = (4, 4), 8
pipe = StableDiffusionPipeline.from_pretrained(REPO, torch_dtype=torch.float32, safety_checker=None)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
pipe = pipe.to("cpu"); pipe.set_progress_bar_config(disable=True)
unet, te, vae, tk = pipe.unet.eval(), pipe.text_encoder.eval(), pipe.vae.eval(), pipe.tokenizer
psnr = lambda a, b: 10 * np.log10(255 ** 2 / (np.mean((a.astype(np.float64) - b) ** 2) + 1e-12))
ssim = lambda a, b: float(SSIM(np.asarray(Image.fromarray(a).convert("L")), np.asarray(Image.fromarray(b).convert("L")), data_range=255))
snr = lambda a, b: 10 * np.log10((a ** 2).mean() / (((a - b) ** 2).mean() + 1e-30))

print("=== 1) EXPORTAR .ttq ===")
def estado(m, tirar=None, so=None):
    d = {}
    for k, v in m.state_dict().items():
        if not v.is_floating_point(): continue
        if tirar and k.startswith(tirar): k = k[len(tirar):]
        if so and not k.startswith(so): continue
        d[k] = v.detach().float().numpy()
    return d
est_vae = estado(vae, so="decoder.")
for nome, est, rec in (("unet", estado(unet), None), ("texto", estado(te, "text_model."), None),
                       ("vae", est_vae, {k: "fp16" for k in est_vae})):
    cam = "sdxs_%s_%s.ttq" % (nome, "fp16" if nome == "vae" else "q8")
    r = cq.salvar(est, cam, BLOCO, BITS, receita=rec)
    print("  %-6s %-18s %7.1f MB  (fp32 %.1f MB)" % (nome, cam, os.path.getsize(cam) / 1e6, r["bruto_fp32_mb"]))
os.makedirs("tok", exist_ok=True); tk.save_pretrained("tok")
n_unet = sum(p.numel() for p in unet.parameters())
print("  UNet em 8 bits 4x4 = %.1f%% do fp16" % (100 * os.path.getsize("sdxs_unet_q8.ttq") / (2 * n_unet)))

print("\n=== 2) TEXTO (CLIP numpy x torch) ===")
tok = pt.TokenizadorCLIP("tok/tokenizer.json")
ids_t = tk(PROMPT, padding="max_length", max_length=77, truncation=True).input_ids
print("  tokenizador igual:", tok.codificar(PROMPT) == ids_t)
with torch.no_grad(): enc_t = te(torch.tensor([ids_t]))[0][0].numpy()
P = mc.Pesos("sdxs_texto_q8.ttq", cache_mb=0); enc_n = cv.clip_texto(ids_t, P); P.fechar()
print("  SNR texto (8 bits x fp32): %.1f dB" % snr(enc_t, enc_n))

print("\n=== 3) UNET (numpy com 8 bits x torch fp32), t=999 ===")
x0 = np.random.RandomState(1).randn(4, 32, 32).astype(np.float32)
with torch.no_grad(): eps_t = unet(torch.tensor(x0)[None], torch.tensor([999]), encoder_hidden_states=torch.tensor(enc_t)[None]).sample[0].numpy()
U = mc.Pesos("sdxs_unet_q8.ttq", cache_mb=0); t0 = time.time()
eps_n = us.unet_sdxs([x0], 999, [enc_t], U)[0]; U.fechar()
print("  SNR do UNet: %.1f dB  (>= 30 = bom) | %.1f s no Colab (32x32)" % (snr(eps_t, eps_n), time.time() - t0))

print("\n=== 4) TAESD (numpy x torch) num latente real ===")
with torch.no_grad():
    lat = pipe(PROMPT, num_inference_steps=1, guidance_scale=0, latents=torch.tensor(x0)[None], height=256, width=256, output_type="latent").images
    img_t = vae.decode(lat).sample[0].numpy()
V = mc.Pesos("sdxs_vae_fp16.ttq", cache_mb=0); img_n = tn.decodificar(lat[0].numpy(), V); V.fechar()
print("  SNR do TAESD: %.1f dB  | faixa torch [%.2f, %.2f], numpy [%.2f, %.2f]" % (snr(img_t, img_n), img_t.min(), img_t.max(), img_n.min(), img_n.max()))

print("\n=== 5) PONTA A PONTA: 1 passo, sem guia (numpy 8 bits x torch fp32, mesmo ruido) ===")
os.makedirs("_p", exist_ok=True)
for a_, b_ in (("sdxs_unet_q8.ttq", "sdxs_unet_q8.ttq"), ("sdxs_texto_q8.ttq", "sdxs_texto_q8.ttq"), ("sdxs_vae_fp16.ttq", "sdxs_vae_fp16.ttq"), ("tok/tokenizer.json", "tokenizer.json")):
    d_ = os.path.join("_p", b_)
    if not os.path.lexists(d_): os.symlink(os.path.abspath(a_), d_)
res = {}
for lado in (256, 512):
    lat0 = np.random.RandomState(7).randn(4, lado // 8, lado // 8).astype(np.float32)
    t0 = time.time()
    with torch.no_grad():
        ref = pipe(PROMPT, num_inference_steps=1, guidance_scale=0, latents=torch.tensor(lat0)[None], height=lado, width=lado, output_type="np").images[0]
    ref = (ref * 255 + 0.5).clip(0, 255).astype("uint8"); tt = time.time() - t0
    t0 = time.time(); im = ps.gerar("_p", PROMPT, lado=lado, latente=lat0, log=lambda *_: None); tn_ = time.time() - t0
    Image.fromarray(ref).save("sdxs_torch_%d.png" % lado); Image.fromarray(im).save("sdxs_numpy_%d.png" % lado)
    res[lado] = {"psnr": float(psnr(ref, im)), "ssim": ssim(ref, im), "t_torch": tt, "t_numpy": tn_}
    print("  %dpx: PSNR %.1f dB  SSIM %.4f  | torch %.1f s, numpy %.1f s (Colab CPU)" % (lado, res[lado]["psnr"], res[lado]["ssim"], tt, tn_))
json.dump(res, open("sdxs_validacao.json", "w"), indent=1)
arqs = ["sdxs_unet_q8.ttq", "sdxs_texto_q8.ttq", "sdxs_vae_fp16.ttq", "tok/tokenizer.json", "sdxs_validacao.json"] + ["sdxs_%s_%d.png" % (a, l) for a in ("torch", "numpy") for l in (256, 512)]
with zipfile.ZipFile("sdxs_cubo.zip", "w", zipfile.ZIP_STORED) as z:
    for a in arqs: z.write(a)
print("\nPRONTO: sdxs_cubo.zip = %.1f MB. Me mande TODO este texto e as imagens (sdxs_torch_*.png e sdxs_numpy_*.png)." % (os.path.getsize("sdxs_cubo.zip") / 1e6))
try:
    from google.colab import files; files.download("sdxs_cubo.zip")
except Exception:
    print("Baixe sdxs_cubo.zip pelo painel de arquivos do Colab.")
'''
open(os.path.join(aqui, "colab_sdxs_validar.py"), "w", encoding="utf-8").write(corpo)
print("ok", max(len(l) for l in corpo.splitlines()))
