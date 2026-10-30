"""CUBO-QUANT K: quantizacao em blocos ASSIMETRICA (minimo + escala) com busca da melhor faixa.

Por que existe: o cubo_quant.py usa so uma escala por cubo (simetrico, max/7). No tiny-sd real isso
quebrou o 4 bits (SSIM <= 0.42). Aqui cada bloco guarda `minimo` e `escala` (float16) e a faixa e
ENCOLHIDA quando isso reduz o erro quadratico do bloco (ignora um ou outro valor extremo), como o
Q4_K/AWQ fazem. Blocos finos (1 x 32, ao longo da entrada) seguem o estilo do GGUF.

So numpy. Custo em bits/peso (4 bits): bloco de 32 valores = 5.0 ; de 64 = 4.5 (escala+minimo em float16).
STATUS: testado em dados sinteticos; o ganho no modelo real so se prova com teste_k_unico (Colab).
"""
import numpy as np

FATORES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65)     # quanto da faixa original manter


def _alinha(n, a):
    return (n + a - 1) // a * a


def _em_blocos(m, bloco):
    bh, bw = bloco
    R, C = m.shape
    Rp, Cp = _alinha(R, bh), _alinha(C, bw)
    p = np.pad(m, ((0, Rp - R), (0, Cp - C)), mode="edge") if (Rp, Cp) != (R, C) else m
    b = p.reshape(Rp // bh, bh, Cp // bw, bw).transpose(0, 2, 1, 3).reshape(-1, bh * bw)
    return b, Rp // bh, Cp // bw


def quantizar_k(arr, bloco=(1, 32), bits=4, busca=True, chunk=1 << 20):
    """Devolve (pack uint8, escala f16, minimo f16, R, C). arr: qualquer forma (linhas x resto)."""
    m = np.asarray(arr, np.float32)
    m = m.reshape(m.shape[0], -1)
    R, C = m.shape
    b, nbh, nbw = _em_blocos(m, bloco)
    nb, n = b.shape
    L = float(2 ** bits - 1)
    fat = FATORES if busca else (1.0,)
    q_all = np.empty((nb, n), np.uint8)
    esc_all = np.empty(nb, np.float16)
    mn_all = np.empty(nb, np.float16)
    por = max(1, chunk // n)
    for a in range(0, nb, por):
        x = b[a:a + por]
        lo, hi = x.min(1), x.max(1)
        mid, meia = (hi + lo) / 2, (hi - lo) / 2
        melhor = np.full(len(x), np.inf, np.float32)
        for f in fat:
            l = (mid - meia * f).astype(np.float16)
            sc = np.where(meia * f > 0, 2 * meia * f / L, 1.0).astype(np.float16)
            l32, s32 = l.astype(np.float32), sc.astype(np.float32)
            s32 = np.where(s32 == 0, 1.0, s32)
            q = np.clip(np.rint((x - l32[:, None]) / s32[:, None]), 0, L)
            e = ((q * s32[:, None] + l32[:, None] - x) ** 2).sum(1)
            k = e < melhor
            if k.any():
                melhor[k] = e[k]
                q_all[a:a + por][k] = q[k].astype(np.uint8)
                esc_all[a:a + por][k] = sc[k]
                mn_all[a:a + por][k] = l[k]
    plano = q_all.reshape(-1)
    if bits == 4:
        if plano.size % 2:
            plano = np.concatenate([plano, np.zeros(1, np.uint8)])
        pack = ((plano[0::2] << 4) | plano[1::2]).astype(np.uint8)
    else:
        pack = plano
    return pack, esc_all, mn_all, R, C


def desquantizar_k(pack, esc, mn, R, C, forma, bloco=(1, 32), bits=4, dtype=np.float32):
    bh, bw = bloco
    nbh, nbw = _alinha(R, bh) // bh, _alinha(C, bw) // bw
    nb, n = nbh * nbw, bh * bw
    if bits == 4:
        plano = np.empty(pack.size * 2, np.uint8)
        plano[0::2], plano[1::2] = pack >> 4, pack & 15
        q = plano[:nb * n]
    else:
        q = pack[:nb * n]
    s = esc.astype(np.float32)
    s = np.where(s == 0, 1.0, s)
    x = q.reshape(nb, n).astype(np.float32) * s[:, None] + mn.astype(np.float32)[:, None]
    x = x.reshape(nbh, nbw, bh, bw).transpose(0, 2, 1, 3).reshape(nbh * bh, nbw * bw)
    return x[:R, :C].reshape(forma).astype(dtype, copy=False)


def bytes_k(pack, esc, mn):
    return pack.nbytes + esc.nbytes + mn.nbytes


def bits_por_peso(bloco, bits):
    return bits + 32.0 / (bloco[0] * bloco[1])
