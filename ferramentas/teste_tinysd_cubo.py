# TTOKEN - BARREDURA DO CUBO-QUANT NO tiny-sd (modelo real). Colab/PC com torch; CPU basta.
# Roda os MESMOS dados do seu teste (seed 5762, 256x256, 4 passos) e compara configuracoes que o
# codec de cores nao testou: cubos pequenos, 4 bits com cubos finos, reordenar, e TODOS os
# pesos (convolucoes tambem), nao so as Linear. Precisa de cubo_quant.py na mesma pasta.
# NAO FOI EXECUTADO AINDA (o sandbox onde escrevi nao tem torch): rode e me mande a tabela.
import os, sys, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate"])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/content")
import numpy as np, torch
import cubo_quant as cq
from diffusers import AutoPipelineForText2Image

SEED, PROMPT, STEPS, SIZE = 5762, "Carro futurista correndo na chuva cyberpunk", 4, 256
pipe = AutoPipelineForText2Image.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
pipe.set_progress_bar_config(disable=True)
if hasattr(pipe, "safety_checker"): pipe.safety_checker = None
gerar = lambda: pipe(prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=7.5,
                     generator=torch.Generator("cpu").manual_seed(SEED), height=SIZE, width=SIZE).images[0]

def psnr(a, b):
    x, y = np.asarray(a.convert("RGB"), np.float64), np.asarray(b.convert("RGB"), np.float64)
    m = np.mean((x - y) ** 2); return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)
def ssim(a, b):                      # SSIM de janela (padrao); cai no global se nao houver skimage
    try:
        from skimage.metrics import structural_similarity as S
        return float(S(np.asarray(a.convert("L")), np.asarray(b.convert("L")), data_range=255))
    except Exception:
        x, y = [np.asarray(i.convert("L"), np.float64) for i in (a, b)]
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2; mx, my = x.mean(), y.mean()
        return float(((2*mx*my+c1)*(2*((x-mx)*(y-my)).mean()+c2))/((mx**2+my**2+c1)*(x.var()+y.var()+c2)))

unet = pipe.unet
alvo = {n: p for n, p in unet.named_parameters() if p.ndim >= 2 and p.numel() >= cq.MIN_ELEMENTOS}
orig = {n: p.data.clone() for n, p in alvo.items()}
fp16_bytes = sum(p.numel() * 2 for p in alvo.values())
print("tensores 2D+ do UNet: %d | %.1f MB em fp16 (convolucoes incluidas)" % (len(alvo), fp16_bytes / 1e6))
base = gerar(); base.save("base.png")

def rodar(rotulo, bits, bloco, reordenar):
    t = time.time(); bytes_q = 0
    for n, p in alvo.items():
        m = orig[n].numpy().reshape(orig[n].shape[0], -1)
        if reordenar:
            pr, pc = cq.ordem_cubo(m); m = m[pr][:, pc]
        pk, es, R, C = cq.quantizar(m, bloco, bits)
        rec = cq.desquantizar(pk, es, R, C, (R, C), bloco, bits=bits)
        if reordenar:
            rec = rec[np.argsort(pr)][:, np.argsort(pc)]
        p.data = torch.from_numpy(rec.reshape(orig[n].shape)).float()
        bytes_q += pk.nbytes + es.nbytes + (8 * (R + C) // 2 if reordenar else 0)
    img = gerar()
    print("%-34s tamanho %5.1f%% do fp16 | PSNR %5.1f dB | SSIM %.4f | %.0fs" %
          (rotulo, 100 * bytes_q / fp16_bytes, psnr(base, img), ssim(base, img), time.time() - t))
    img.save("cfg_%s.png" % rotulo.replace(" ", "_").replace("/", "-"))
    for n, p in alvo.items(): p.data = orig[n].clone()

print("\nconfiguracao                        resultado")
rodar("8 bits, cubo 8x8", 8, (8, 8), False)
rodar("8 bits, cubo 64x64 (igual ao seu)", 8, (64, 64), False)
rodar("4 bits, cubo 64x64 (igual ao seu)", 4, (64, 64), False)
rodar("4 bits, cubo 8x8", 4, (8, 8), False)
rodar("4 bits, cubo 4x4", 4, (4, 4), False)
rodar("4 bits, cubo 1x32 (estilo Q4_0)", 4, (1, 32), False)
rodar("4 bits, cubo 8x8 reordenado", 4, (8, 8), True)
rodar("4 bits, cubo 1x32 reordenado", 4, (1, 32), True)
