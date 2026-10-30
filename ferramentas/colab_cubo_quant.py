# TTOKEN - TESTE DO CUBO-QUANT NO COLAB (GPU T4). Cole TUDO em UMA celula e rode.
# Antes: suba o arquivo cubo_quant.py no painel de arquivos do Colab (pasta /content),
# ou rode:  !wget -q https://raw.githubusercontent.com/fabiopimentapinto16-ai/ttokem-studio-apk/main/ferramentas/cubo_quant.py
# NAO FOI TESTADO COM O MODELO REAL (o sandbox onde escrevi nao tem torch nem internet):
# o teste com pesos sinteticos passa; aqui voce mede a qualidade REAL da imagem.
import os, sys, time
import numpy as np
os.system("pip install -q diffusers transformers accelerate")
sys.path.insert(0, "/content")
import torch
import cubo_quant as cq
from diffusers import AutoPipelineForText2Image

assert torch.cuda.is_available(), "Ative a GPU: Ambiente de execucao > Alterar tipo > T4 GPU"
pipe = AutoPipelineForText2Image.from_pretrained("IDKiro/sdxs-512-0.9", torch_dtype=torch.float16).to("cuda")
PROMPT = "A majestic red fox standing in the snow at night, photorealistic, detailed fur"
gerar = lambda: pipe(PROMPT, num_inference_steps=1, guidance_scale=0.0,
                     generator=torch.Generator("cuda").manual_seed(42)).images[0]

original = gerar()                                          # imagem com os pesos originais

BLOCO = (8, 8)          # se a imagem ficar ruim, tente (4, 4) ou (1, 32)
estado = {k: v.float().cpu().numpy() for k, v in pipe.unet.state_dict().items()}
t = time.time()
info = cq.salvar(estado, "/content/sdxs_unet.ttq", BLOCO)
print("quantizado em %.1fs:" % (time.time() - t), info)

pesos_orig = {k: v.clone() for k, v in pipe.unet.state_dict().items()}
sd = cq.para_torch("/content/sdxs_unet.ttq", "cuda", "float16")
pipe.unet.load_state_dict(sd, strict=True)                  # strict: avisa se alguma chave faltar
quant = gerar()

a = np.asarray(original, np.float32); b = np.asarray(quant, np.float32)
mse = float(((a - b) ** 2).mean()); psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
print("PSNR original x cubo: %.1f dB  (acima de ~30 dB e quase igual; abaixo de ~20 dB e visivel)" % psnr)
from IPython.display import display
from PIL import Image
lado = Image.new("RGB", (original.width * 2, original.height))
lado.paste(original, (0, 0)); lado.paste(quant, (original.width, 0)); display(lado)
lado.save("/content/comparacao_original_x_cubo.png")
