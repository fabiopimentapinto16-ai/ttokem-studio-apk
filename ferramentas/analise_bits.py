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
    """Devolve (bits_por_peso, ruido, sinal) para uma matriz m (R,C)."""
    b, (R, C) = _blocos(m, g)
    rot = metodo == "hadamard"
    if rot:
        H = hadamard(g)
        b = b @ H.T
    q, rec = quantizar_bloco(b, bits)
    if rot:
        rec = rec @ H
    rec, b = rec.reshape(R, -1)[:, :C], b.reshape(R, -1)[:, :C]
    ruido, sinal = float(((m - rec) ** 2).sum()), float((m ** 2).sum())
    custo = (bits if metodo == "base" else entropia_bits(q)) + 16.0 / g
    return custo, ruido, sinal


def varrer(tensores, grade, metodos=("base", "entropia", "hadamard")):
    """tensores: dict nome->ndarray (>=2D). grade: lista de (bits, g).
    Devolve linhas (metodo, bits, g, bits_por_peso, snr_db, vezes_menor_que_fp16)."""
    linhas = []
    for metodo in metodos:
        for bits, g in grade:
            tot_bits = tot_n = tot_r = tot_s = 0.0
            for arr in tensores.values():
                m = np.asarray(arr, np.float32).reshape(arr.shape[0], -1)
                custo, r, s = avaliar(m, bits, g, metodo)
                tot_bits += custo * m.size; tot_n += m.size; tot_r += r; tot_s += s
            bpp = tot_bits / tot_n
            snr = 10 * np.log10(tot_s / max(tot_r, 1e-30))
            linhas.append((metodo, bits, g, bpp, snr, 16.0 / bpp))
    return linhas
