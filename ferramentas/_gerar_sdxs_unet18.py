"""Gera colab_sdxs_unet18.py: exporta SO o UNet do SDXS em 8 bits 1x8 (melhor fidelidade) para trocar no celular."""
import os, json, base64, zlib
aqui = os.path.dirname(os.path.abspath(__file__))
src = {"cubo_quant": open(os.path.join(aqui, "cubo_quant.py"), encoding="utf-8").read()}
b64 = base64.b64encode(zlib.compress(json.dumps(src).encode("utf-8"), 9)).decode()
emb = "_PAC = (\n" + "\n".join('    "%s"' % b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n)\n"
corpo = r'''# TTOKEN - UNet do SDXS em 8 bits 1x8 (SSIM 0,931 contra 0,911 do 4x4). Sem GPU. Gera sdxs_unet_q8.ttq (~395 MB).
# Depois: baixe o arquivo e COLOQUE NA PASTA sdxs_cubo do celular, substituindo o sdxs_unet_q8.ttq antigo.
import os, sys, types, json, zlib, base64, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate"])
import numpy as np, torch
''' + emb + r'''
_SRC = json.loads(zlib.decompress(base64.b64decode(_PAC)).decode("utf-8"))
cq = types.ModuleType("cubo_quant"); sys.modules["cubo_quant"] = cq; exec(_SRC["cubo_quant"], cq.__dict__)
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("IDKiro/sdxs-512-dreamshaper", subfolder="unet", torch_dtype=torch.float32).eval()
est = {k: v.detach().float().numpy() for k, v in unet.state_dict().items() if v.is_floating_point()}
r = cq.salvar(est, "sdxs_unet_q8.ttq", (1, 8), 8)
mb = os.path.getsize("sdxs_unet_q8.ttq") / 1e6
print("sdxs_unet_q8.ttq: %.1f MB (esperado ~395 MB, 8 bits 1x8)" % mb)
try:
    from google.colab import files; files.download("sdxs_unet_q8.ttq")
except Exception:
    print("Baixe sdxs_unet_q8.ttq pelo painel de arquivos do Colab.")
'''
open(os.path.join(aqui, "colab_sdxs_unet18.py"), "w", encoding="utf-8").write(corpo)
print("ok")
