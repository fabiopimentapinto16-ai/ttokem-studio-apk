# TTOKEN - PASSO 1 DO SDXS-512 (imagem em 1 passo, sem guia): descobrir a arquitetura para adaptar o motor numpy.
# Nao precisa de GPU. Gera uma imagem de teste, imprime a arquitetura do UNet/VAE e salva um arquivo com tudo.
# Depois: me mande TODO o texto impresso + sdxs_arquitetura.json + sdxs_teste.png.
import os, sys, json, subprocess, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate"])
import numpy as np, torch
from diffusers import StableDiffusionPipeline, AutoencoderTiny

REPO = "IDKiro/sdxs-512-dreamshaper"
dev = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(REPO, torch_dtype=torch.float32)
try:
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
    vae_nome = "madebyollin/taesd"
except Exception as ex:
    vae_nome = "(do proprio repo) %s" % ex
pipe = pipe.to(dev); pipe.set_progress_bar_config(disable=True)
if hasattr(pipe, "safety_checker"): pipe.safety_checker = None

u, te, vae = pipe.unet, pipe.text_encoder, pipe.vae
cfg = dict(u.config)
chaves = ["sample_size", "in_channels", "out_channels", "down_block_types", "up_block_types", "mid_block_type", "block_out_channels",
          "layers_per_block", "cross_attention_dim", "attention_head_dim", "num_attention_heads", "transformer_layers_per_block",
          "use_linear_projection", "only_cross_attention", "norm_num_groups", "act_fn", "time_embedding_type", "flip_sin_to_cos",
          "freq_shift", "resnet_time_scale_shift", "downsample_padding", "norm_eps", "dual_cross_attention", "upcast_attention"]
print("=" * 70); print("UNET  (%s)  params: %.1f M" % (type(u).__name__, sum(p.numel() for p in u.parameters()) / 1e6))
for k in chaves:
    if k in cfg: print("  %-30s %s" % (k, cfg[k]))
print("  mid_block (objeto):", type(u.mid_block).__name__ if getattr(u, "mid_block", None) is not None else None)
print("\nBLOCOS DO UNET (tipo, resnets, atencoes, transformer_blocks por atencao, so cross-attn?):")
def descrever(nome, blocos):
    for i, b in enumerate(blocos):
        att = getattr(b, "attentions", None)
        tb = len(att[0].transformer_blocks) if att else 0
        extra = ""
        if att:
            a1 = att[0].transformer_blocks[0]
            extra = " attn1.heads=%s  ff=%s" % (a1.attn1.heads, type(a1.ff.net[0]).__name__)
        print("  %s[%d] %-24s resnets=%d  atencoes=%d  tb/atencao=%d  down/up-sample=%s%s" % (
            nome, i, type(b).__name__, len(b.resnets), len(att) if att else 0, tb,
            bool(getattr(b, "downsamplers", None) or getattr(b, "upsamplers", None)), extra))
descrever("down", u.down_blocks); descrever("up", u.up_blocks)
print("\nVAE: %s (%s) params %.2f M" % (type(vae).__name__, vae_nome, sum(p.numel() for p in vae.parameters()) / 1e6))
vc = dict(vae.config)
for k in ["in_channels", "out_channels", "latent_channels", "encoder_block_out_channels", "decoder_block_out_channels", "num_blocks",
          "act_fn", "upsampling_scaling_factor", "latent_magnitude", "latent_shift", "scaling_factor", "block_out_channels",
          "layers_per_block"]:
    if k in vc: print("  %-30s %s" % (k, vc[k]))
print("\nTEXTO: %s  params %.1f M  camadas=%s  ocultas=%s" % (type(te).__name__, sum(p.numel() for p in te.parameters()) / 1e6,
      te.config.num_hidden_layers, te.config.hidden_size))
print("\nAGENDADOR:", type(pipe.scheduler).__name__, dict(pipe.scheduler.config))

# nomes e formas de todos os pesos (para o conversor)
def formas(m): return {n: list(p.shape) for n, p in m.named_parameters()}
arq = {"unet_config": {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v)) for k, v in cfg.items()},
       "vae_config": {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v)) for k, v in vc.items()},
       "unet_pesos": formas(u), "vae_pesos": formas(vae), "agendador": {k: str(v) for k, v in pipe.scheduler.config.items()}}
json.dump(arq, open("sdxs_arquitetura.json", "w"), indent=1)

# imagem de teste em 1 passo, sem guia + tempo
PROMPT = "a red apple on a wooden table"
for tam in (256, 512):
    t0 = time.time()
    im = pipe(prompt=PROMPT, num_inference_steps=1, guidance_scale=0, height=tam, width=tam,
              generator=torch.Generator("cpu").manual_seed(42)).images[0]
    print("\nimagem %dx%d em 1 passo: %.1f s  (%s)" % (tam, tam, time.time() - t0, dev))
    im.save("sdxs_teste_%d.png" % tam)

# entrada/saida de referencia do UNet (para validar o motor numpy depois)
with torch.no_grad():
    ids = pipe.tokenizer(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(dev)
    enc = te(ids)[0]
    x = torch.from_numpy(np.random.RandomState(1).randn(1, 4, 32, 32).astype(np.float32)).to(dev)
    t = torch.tensor([999], device=dev)
    eps = u(x, t, encoder_hidden_states=enc).sample
np.savez_compressed("sdxs_referencia_unet.npz", x=x.cpu().numpy(), t=np.array([999]), enc=enc.cpu().numpy(), eps=eps.cpu().numpy())
print("\nSalvos: sdxs_arquitetura.json, sdxs_teste_256.png, sdxs_teste_512.png, sdxs_referencia_unet.npz")
try:
    from google.colab import files
    for f in ("sdxs_arquitetura.json", "sdxs_teste_256.png", "sdxs_teste_512.png", "sdxs_referencia_unet.npz"): files.download(f)
except Exception:
    pass
print("Me mande TODO este texto e os arquivos baixados.")
