"""PIPELINE OFFLINE SDXS-512 em Python puro + numpy: texto -> CLIP -> UNet (1 PASSO, sem guia) -> TAESD -> PNG.
Pesos (pasta): sdxs_unet_q8.ttq (ou sdxs_unet_curva.ttq), sdxs_texto_q8.ttq, sdxs_vae_fp16.ttq, tokenizer.json.
Uso:  python pipeline_sdxs.py PASTA "a red apple on a table" --lado 256 --semente 1 --saida maca.png
Uma unica chamada do UNet (o tiny-sd gasta 8): em tese ~8x menos conta por imagem. Resolucao: lado multiplo de 8 (256 ou 512).
"""
import os
import sys
import time
import argparse
import numpy as np

try:
    _AQUI = os.path.dirname(os.path.abspath(__file__))
    if _AQUI not in sys.path:
        sys.path.insert(0, _AQUI)
except NameError:
    pass
import motor_cubico as mc            # noqa: E402
import clip_vae_tinysd as cv         # noqa: E402
import pipeline_tinysd as pt         # noqa: E402
import unet_sdxs as us               # noqa: E402
import taesd_numpy as tn             # noqa: E402


def gerar(pasta, prompt, semente=1, lado=256, latente=None, log=print):
    t0 = time.time()
    tok = pt.TokenizadorCLIP(pt._achar(pasta, "tokenizer.json", ["tokenizer.json", "tok/tokenizer.json"]))
    f_unet = pt._achar(pasta, "UNet", ["sdxs_unet_curva.ttq", "sdxs_unet_q8.ttq"])
    f_txt = pt._achar(pasta, "texto", ["sdxs_texto_q8.ttq"])
    f_vae = pt._achar(pasta, "vae", ["sdxs_vae_fp16.ttq"])
    P = mc.Pesos(f_txt, cache_mb=0)
    cond = cv.clip_texto(tok.codificar(prompt), P)
    P.fechar()
    log("texto ok (%.1f s)" % (time.time() - t0))
    ag = pt.AgendadorDPM()
    ag.definir_passos(1)                                           # 1 passo: t = 999, saida = x0
    if latente is None:
        latente = np.random.RandomState(semente).randn(4, lado // 8, lado // 8).astype(np.float32)
    x = latente.astype(np.float32)
    U = mc.Pesos(f_unet, cache_mb=0)
    ts = time.time()
    eps = us.unet_sdxs([x], int(ag.timesteps[0]), [cond], U)[0]
    x = ag.passo(eps, x)
    U.fechar()
    log("UNet 1 passo (%.1f s)" % (time.time() - ts))
    V = mc.Pesos(f_vae, cache_mb=0)
    img = cv.para_imagem(tn.decodificar(x, V))
    V.fechar()
    log("pronto em %.1f s" % (time.time() - t0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("prompt")
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--lado", type=int, default=256)
    ap.add_argument("--saida", default="saida_sdxs.png")
    a = ap.parse_args()
    pt.salvar_png(gerar(a.pasta, a.prompt, a.semente, a.lado), a.saida)
    print("salvo", a.saida)


if __name__ == "__main__":
    main()
