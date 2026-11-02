"""Quantos BITS POR PESO cada tecnica gasta para chegar a mesma fidelidade (SNR nos pesos).

Tecnicas (todas com blocos 1 x g ao longo das colunas, escala fp16 por bloco, simetrico):
  base      : inteiro de b bits, tamanho = b + 16/g bits/peso (o que o .ttq faz, com bloco menor)
  entropia  : mesmos inteiros, mas contados pela ENTROPIA empirica (limite de um codificador Huffman/aritmetico)
  hadamard  : rotacao de Hadamard (g x g) em cada bloco ANTES de quantizar (espalha valores extremos), + entropia
A rotacao e ortogonal: a leitura aplica a inversa, custo O(g log g) por bloco. Medida = SNR nos pesos (proxy;
a fidelidade da IMAGEM continua sendo decidida pelo teste com UNet).
"""
import numpy as np


def hadamard(n):
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _blocos(m, g):
    R, C = m.shape
    Cp = (C + g - 1) // g * g
    p = np.zeros((R, Cp), np.float32); p[:, :C] = m
    return p.reshape(R, Cp // g, g), (R, C)


def quantizar_bloco(b, bits):
    """b: (..., g). Devolve (inteiros, reconstrucao). Escala = absmax/nivel, arredondada para fp16."""
    nivel = 2 ** (bits - 1) - 1
    mx = np.abs(b).max(-1, keepdims=True)
    esc = np.where(mx == 0, 1.0, mx / nivel).astype(np.float16).astype(np.float32)
    q = np.clip(np.rint(b / esc), -nivel - 1, nivel)
    return q.astype(np.int16), q * esc


def entropia_bits(q):
    """Entropia empirica (bits/simbolo) dos inteiros q."""
    v, c = np.unique(q, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def avaliar(m, bits, g, metodo):
    """Devolve (bits_por_peso, ruido, sinal) para uma matriz m (R,C). Mantida para compatibilidade."""
    r = avaliar_ambos(m, bits, g, rot=(metodo == "hadamard"))
    custo = r["base"] if metodo == "base" else r["entropia"]
    return custo, r["ruido"], r["sinal"]


def _entropia_rapida(q, bits):
    """Entropia empirica via bincount (muito mais rapido que np.unique)."""
    c = np.bincount((q.reshape(-1) + 2 ** (bits - 1)).astype(np.int64), minlength=2 ** bits).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def avaliar_ambos(m, bits, g, rot=False):
    """Quantiza UMA vez e devolve custo 'base' (bits fixos), custo 'entropia', ruido e sinal."""
    b, (R, C) = _blocos(m, g)
    if rot:
        H = hadamard(g)
        b = b @ H.T
    q, rec = quantizar_bloco(b, bits)
    if rot:
        rec = rec @ H
        b = b @ H
    rec, b = rec.reshape(R, -1)[:, :C], b.reshape(R, -1)[:, :C]
    esc = 16.0 / g
    return {"base": bits + esc, "entropia": _entropia_rapida(q, bits) + esc,
            "ruido": float(((m - rec) ** 2).sum()), "sinal": float((m ** 2).sum())}


def varrer(tensores, grade, metodos=("base", "entropia", "hadamard"), max_pesos=None, semente=0):
    """tensores: dict nome->ndarray (>=2D). grade: lista de (bits, g).
    max_pesos: se dado, usa uma AMOSTRA de linhas (mesma fracao em todos os tensores) para ir mais rapido.
    Devolve linhas (metodo, bits, g, bits_por_peso, snr_db, vezes_menor_que_fp16)."""
    total = sum(np.asarray(a).size for a in tensores.values())
    frac = 1.0 if not max_pesos or total <= max_pesos else max_pesos / total
    rng = np.random.RandomState(semente)
    mats = []
    for arr in tensores.values():
        m = np.asarray(arr, np.float32).reshape(arr.shape[0], -1)
        if frac < 1.0:
            n = max(1, int(round(m.shape[0] * frac)))
            m = m[np.sort(rng.choice(m.shape[0], n, replace=False))]
        mats.append(m)
    linhas = []
    for bits, g in grade:
        for rot in (False, True):
            if rot and "hadamard" not in metodos:
                continue
            if not rot and not ({"base", "entropia"} & set(metodos)):
                continue
            acc = {"base": 0.0, "entropia": 0.0, "ruido": 0.0, "sinal": 0.0}
            n = 0
            for m in mats:
                r = avaliar_ambos(m, bits, g, rot)
                for k in ("base", "entropia"):
                    acc[k] += r[k] * m.size
                acc["ruido"] += r["ruido"]; acc["sinal"] += r["sinal"]; n += m.size
            snr = 10 * np.log10(acc["sinal"] / max(acc["ruido"], 1e-30))
            if not rot:
                for nome in ("base", "entropia"):
                    if nome in metodos:
                        bpp = acc[nome] / n
                        linhas.append((nome, bits, g, bpp, snr, 16.0 / bpp))
            else:
                bpp = acc["entropia"] / n
                linhas.append(("hadamard", bits, g, bpp, snr, 16.0 / bpp))
    return linhas
