# TTOKEN - UNet do SDXS em 8 bits 1x8 (SSIM 0,931 contra 0,911 do 4x4). Sem GPU. Gera sdxs_unet_q8.ttq (~395 MB).
# Salva no seu Google Drive (Meu Drive > sdxs_cubo). Depois baixe pelo app Drive e COLOQUE NA PASTA sdxs_cubo do celular.
import os, sys, types, json, zlib, base64, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate"])
import numpy as np, torch
_PAC = (
    "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"
    "Bzq9L/8CDNUhwQ=="
)

_SRC = json.loads(zlib.decompress(base64.b64decode(_PAC)).decode("utf-8"))
cq = types.ModuleType("cubo_quant"); sys.modules["cubo_quant"] = cq; exec(_SRC["cubo_quant"], cq.__dict__)
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("IDKiro/sdxs-512-dreamshaper", subfolder="unet", torch_dtype=torch.float32).eval()
est = {k: v.detach().float().numpy() for k, v in unet.state_dict().items() if v.is_floating_point()}
r = cq.salvar(est, "sdxs_unet_q8.ttq", (1, 8), 8)
mb = os.path.getsize("sdxs_unet_q8.ttq") / 1e6
print("sdxs_unet_q8.ttq: %.1f MB (esperado ~395 MB, 8 bits 1x8)" % mb)
import shutil
try:                                    # download direto de 395 MB costuma travar: salva no Google Drive
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs("/content/drive/MyDrive/sdxs_cubo", exist_ok=True)
    shutil.copy("sdxs_unet_q8.ttq", "/content/drive/MyDrive/sdxs_cubo/sdxs_unet_q8.ttq")
    print("PRONTO: salvo no seu Google Drive em  Meu Drive > sdxs_cubo > sdxs_unet_q8.ttq")
    print("No celular: abra o app Google Drive, toque nos 3 pontinhos do arquivo > Fazer download.")
except Exception as e:
    print("Nao deu para usar o Drive (%s). Baixe pelo painel de arquivos do Colab (icone de pasta)." % e)
