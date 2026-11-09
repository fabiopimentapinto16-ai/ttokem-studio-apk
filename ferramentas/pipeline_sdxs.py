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


_MANTIDOS = {}      # pesos mantidos na RAM entre imagens (videos com varias cenas); so com cache_mb > 0


def _pesos(caminho, cache_mb):
    """Com cache_mb > 0 reaproveita o leitor (e os pesos ja convertidos) entre imagens; senao le e solta."""
    if not cache_mb:
        return mc.Pesos(caminho, cache_mb=0)
    p = _MANTIDOS.get(caminho)
    if p is None:
        p = _MANTIDOS[caminho] = mc.Pesos(caminho, cache_mb=cache_mb)
    return p


def _soltar(p, cache_mb):
    if not cache_mb:
        p.fechar()


def liberar_cache():
    for p in list(_MANTIDOS.values()):
        try:
            p.fechar()
        except Exception:
            pass
    _MANTIDOS.clear()


def gerar(pasta, prompt, semente=1, lado=256, latente=None, log=print, cache_mb=0):
    """cache_mb > 0: mantem os pesos prontos na RAM entre chamadas (so vale a pena com muita RAM livre)."""
    t0 = time.time()
    mc.configurar_hilos(log=log)                    # mede e liga varios nucleos so se ajudar (APK); no Pydroid fica 1
    tok = pt.TokenizadorCLIP(pt._achar(pasta, "tokenizer.json", ["tokenizer.json", "tok/tokenizer.json"]))
    f_unet = pt._achar(pasta, "UNet", ["sdxs_unet_curva.ttq", "sdxs_unet_q8.ttq"])
    f_txt = pt._achar(pasta, "texto", ["sdxs_texto_q8.ttq"])
    f_vae = pt._achar(pasta, "vae", ["sdxs_vae_fp16.ttq"])
    P = _pesos(f_txt, cache_mb)
    cond = cv.clip_texto(tok.codificar(prompt), P)
    _soltar(P, cache_mb)
    log("texto ok (%.1f s)" % (time.time() - t0))
    ag = pt.AgendadorDPM()
    ag.definir_passos(1)                                           # 1 passo: t = 999, saida = x0
    if latente is None:
        latente = np.random.RandomState(semente).randn(4, lado // 8, lado // 8).astype(np.float32)
    x = latente.astype(np.float32)
    U = _pesos(f_unet, cache_mb)
    ts = time.time()
    eps = us.unet_sdxs([x], int(ag.timesteps[0]), [cond], U)[0]
    x = ag.passo(eps, x)
    _soltar(U, cache_mb)
    log("UNet 1 passo (%.1f s)" % (time.time() - ts))
    tn.escolher_conv(log)
    tv = time.time()
    V = _pesos(f_vae, cache_mb)
    img = cv.para_imagem(tn.decodificar(x, V))
    log("VAE (%.1f s)" % (time.time() - tv))
    _soltar(V, cache_mb)
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
