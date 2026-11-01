"""PIPELINE OFFLINE tiny-sd em Python puro + numpy (sem torch, sem internet).

texto -> tokenizador BPE (CLIP) -> CLIP -> UNet (4 passos DPM-Solver++ 2M, CFG) -> VAE -> PNG.
Pesos: 3 arquivos .ttq (cubos quantizados, lidos em streaming) + tokenizer.json.

Uso (Pydroid ou PC), pasta com tinysd_unet_*.ttq, tinysd_texto_q8.ttq, tinysd_vae_q8.ttq, tokenizer.json:
    python pipeline_tinysd.py PASTA "a red apple on a table" --passos 4 --semente 1 --saida maca.png
"""
import os
import re
import sys
import json
import time
import zlib
import struct
import argparse
import numpy as np

try:
    _AQUI = os.path.dirname(os.path.abspath(__file__))
    if _AQUI not in sys.path:
        sys.path.insert(0, _AQUI)
except NameError:                    # embutido num arquivo unico (Colab): os modulos ja estao em sys.modules
    pass
import motor_cubico as mc            # noqa: E402
import unet_tinysd as ut             # noqa: E402
import clip_vae_tinysd as cv         # noqa: E402


# ------------------------------------------------------------------ tokenizador CLIP (BPE de bytes)
def _bytes_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# \p{L}, \p{N} nao existem no `re` padrao: aproximacao por classes de \w.
_PAD = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:_|[^\w\s])+")


class TokenizadorCLIP:
    def __init__(self, caminho_json):
        j = json.load(open(caminho_json, encoding="utf-8"))
        m = j["model"]
        self.vocab = m["vocab"]
        self.rank = {}
        for i, par in enumerate(m["merges"]):
            self.rank[tuple(par) if isinstance(par, list) else tuple(par.split(" "))] = i
        self.sufixo = m.get("end_of_word_suffix", "</w>")
        self.bos, self.eos = self.vocab["<|startoftext|>"], self.vocab["<|endoftext|>"]
        self.b2u = _bytes_unicode()
        self._cache = {}

    def _bpe(self, palavra):
        if palavra in self._cache:
            return self._cache[palavra]
        p = list(palavra[:-1]) + [palavra[-1] + self.sufixo]
        while len(p) > 1:
            melhor, ri = None, 10 ** 9
            for a, b in zip(p, p[1:]):
                r = self.rank.get((a, b))
                if r is not None and r < ri:
                    melhor, ri = (a, b), r
            if melhor is None:
                break
            novo, i = [], 0
            while i < len(p):
                if i < len(p) - 1 and (p[i], p[i + 1]) == melhor:
                    novo.append(p[i] + p[i + 1])
                    i += 2
                else:
                    novo.append(p[i])
                    i += 1
            p = novo
        self._cache[palavra] = p
        return p

    def codificar(self, texto, n=77):
        t = re.sub(r"\s+", " ", texto).strip().lower()
        ids = [self.bos]
        for w in _PAD.findall(t):
            w = "".join(self.b2u[b] for b in w.encode("utf-8"))
            for pedaco in self._bpe(w):
                ids.append(self.vocab.get(pedaco, self.eos))
        ids = ids[:n - 1] + [self.eos]
        return ids + [self.eos] * (n - len(ids))


# ------------------------------------------------------------------ agendador DPM-Solver++ 2M (midpoint)
class AgendadorDPM:
    """Igual ao DPMSolverMultistepScheduler da config do tiny-sd (epsilon, linspace, lower_order_final)."""

    def __init__(self, beta_ini=0.00085, beta_fim=0.012, T=1000):
        b = np.linspace(beta_ini ** 0.5, beta_fim ** 0.5, T, dtype=np.float64) ** 2
        self.ac = np.cumprod(1.0 - b)
        self.T = T

    def definir_passos(self, n):
        ts = np.linspace(0, self.T - 1, n + 1).round()[::-1][:-1].astype(np.int64)
        sig = ((1 - self.ac) / self.ac) ** 0.5
        s = np.interp(ts, np.arange(self.T), sig)
        self.timesteps = ts
        self.sigmas = np.concatenate([s, [0.0]])
        self.saidas = []
        self.n = n
        self.i = 0

    @staticmethod
    def _as(sigma):
        a = 1.0 / np.sqrt(sigma ** 2 + 1)
        return a, sigma * a

    def passo(self, eps, amostra):
        i, n, S = self.i, self.n, self.sigmas
        a_s, s_s = self._as(S[i])
        x0 = (amostra - s_s * eps) / a_s
        self.saidas = (self.saidas + [x0])[-2:]
        a_t, s_t = self._as(S[i + 1])
        ultimo = i == n - 1
        if s_t == 0.0:                               # sigma final 0: devolve x0
            saida = x0
        elif i == 0 or ultimo or len(self.saidas) < 2:
            h = (np.log(a_t) - np.log(s_t)) - (np.log(a_s) - np.log(s_s))
            saida = (s_t / s_s) * amostra - a_t * (np.exp(-h) - 1.0) * x0
        else:
            a_p, s_p = self._as(S[i - 1])
            ls, lt, lp = np.log(a_s) - np.log(s_s), np.log(a_t) - np.log(s_t), np.log(a_p) - np.log(s_p)
            h, h0 = lt - ls, ls - lp
            D1 = (1.0 / (h0 / h)) * (self.saidas[-1] - self.saidas[-2])
            saida = (s_t / s_s) * amostra - a_t * (np.exp(-h) - 1.0) * self.saidas[-1] \
                - 0.5 * a_t * (np.exp(-h) - 1.0) * D1
        self.i += 1
        return saida.astype(np.float32)


# ------------------------------------------------------------------ PNG sem PIL
def salvar_png(rgb, caminho):
    h, w, _ = rgb.shape
    cru = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def ck(tipo, d):
        c = struct.pack(">I", len(d)) + tipo + d
        return c + struct.pack(">I", zlib.crc32(tipo + d) & 0xffffffff)
    with open(caminho, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + ck(b"IDAT", zlib.compress(cru, 6)) + ck(b"IEND", b""))


# ------------------------------------------------------------------ geracao
def _achar(pasta, prefixo, preferidos):
    for nome in preferidos:
        c = os.path.join(pasta, nome)
        if os.path.exists(c):
            return c
    raise FileNotFoundError("nao achei %s em %s (procurei %s)" % (prefixo, pasta, preferidos))


def gerar(pasta, prompt, passos=4, guia=7.5, semente=1, lado=256, negativo="", latente=None, log=print):
    t0 = time.time()
    tok = TokenizadorCLIP(_achar(pasta, "tokenizer.json", ["tokenizer.json", "tok/tokenizer.json"]))
    f_unet = _achar(pasta, "UNet", ["tinysd_unet_curva.ttq", "tinysd_unet_final.ttq", "tinysd_unet_q8.ttq"])
    f_txt = _achar(pasta, "texto", ["tinysd_texto_q8.ttq"])
    f_vae = _achar(pasta, "vae", ["tinysd_vae_q8.ttq"])

    P = mc.Pesos(f_txt, cache_mb=0)                       # 1) texto (libera depois)
    cond = cv.clip_texto(tok.codificar(prompt), P)
    inc = cv.clip_texto(tok.codificar(negativo), P)
    P.fechar()
    log("texto ok (%.1f s)" % (time.time() - t0))

    ag = AgendadorDPM()
    ag.definir_passos(passos)
    if latente is None:
        latente = np.random.RandomState(semente).randn(4, lado // 8, lado // 8).astype(np.float32)
    x = latente.astype(np.float32)
    U = mc.Pesos(f_unet, cache_mb=0)                      # 2) UNet em streaming
    for k, t in enumerate(ag.timesteps):
        ts = time.time()
        eps_u, eps_c = ut.unet_tinysd([x, x], int(t), [inc, cond], U)
        eps = eps_u + guia * (eps_c - eps_u)
        x = ag.passo(eps, x)
        log("passo %d/%d t=%d (%.1f s)" % (k + 1, passos, t, time.time() - ts))
    U.fechar()

    V = mc.Pesos(f_vae, cache_mb=0)                       # 3) VAE
    img = cv.para_imagem(cv.vae_decodificar(x / 0.18215, V))
    V.fechar()
    log("pronto em %.1f s" % (time.time() - t0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("prompt")
    ap.add_argument("--passos", type=int, default=4)
    ap.add_argument("--guia", type=float, default=7.5)
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--lado", type=int, default=256)
    ap.add_argument("--saida", default="saida.png")
    a = ap.parse_args()
    img = gerar(a.pasta, a.prompt, a.passos, a.guia, a.semente, a.lado)
    salvar_png(img, a.saida)
    print("salvo", a.saida)


if __name__ == "__main__":
    main()
