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
