# TTOKEN - SDXS-512 (1 PASSO, SEM GUIA): exporta os pesos em .ttq e VALIDA o motor numpy contra o torch. Arquivo unico.
# Mede: texto (numpy x torch), UNet (8 bits x fp32), TAESD, imagem completa em 256 e 512 px (PSNR/SSIM) e tempo no Colab.
# Gera: sdxs_unet_q8.ttq, sdxs_texto_q8.ttq, sdxs_vae_fp16.ttq, tok/tokenizer.json (+ baixa um .zip com tudo).
import os, sys, types, json, time, zlib, base64, subprocess, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate", "scikit-image"])
import numpy as np, torch
_PAC = (
    "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"
)

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
